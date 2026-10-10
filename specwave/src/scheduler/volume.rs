//! The disk volume that holds a path, with what else decides how fast it reads.

use std::io::Error;
use std::path::Path;

/// A disk volume, with the drive under it, the chip and the operating system's build, which together decide how fast
/// the volume reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Volume {
    /// The volume's universally unique identifier (UUID), eg. "2A8CF686-E3D1-4548-BFFF-59918B6B966D".
    pub uuid: String,
    /// The drive's model, eg. "APPLE SSD AP0512Z", or "" when the system doesn't say, eg. for a network volume.
    pub drive: String,
    /// The processor, eg. "Apple M3 Pro".
    pub chip: String,
    /// The operating system's build, eg. "25G241".
    pub os_build: String,
}

impl Volume {
    /// The volume that holds `path`, which must exist. On macOS the drive's model comes from the IO registry, the
    /// system's tree of devices. A volume without a UUID, and every volume on other systems, gives `Unsupported`.
    pub fn of(path: &Path) -> Result<Volume, Error> {
        imp::of(path)
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::{CStr, CString, c_char, c_void};
    use std::io::{Error, ErrorKind};
    use std::mem::MaybeUninit;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::ptr::{null, null_mut};

    use super::Volume;

    /// A Core Foundation object, eg. a string or a dictionary.
    type CfType = *const c_void;

    /// kCFStringEncodingUTF8.
    const UTF8: u32 = 0x0800_0100;

    /// kIORegistryIterateRecursively and kIORegistryIterateParents, which make a search walk up through every parent.
    const RECURSIVELY: u32 = 1;
    const PARENTS: u32 = 2;

    // SAFETY: these match <CoreFoundation/CoreFoundation.h> and <IOKit/IOKitLib.h>, with every Core Foundation object
    // as a pointer, CFIndex as isize, CFTypeID as usize, Boolean as u8, and mach_port_t and io_object_t as u32. Both
    // frameworks ship with macOS, and the libc crate declares neither.
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(allocator: CfType, text: *const c_char, encoding: u32) -> CfType;
        fn CFStringGetCString(string: CfType, buffer: *mut c_char, size: isize, encoding: u32) -> u8;
        fn CFStringGetTypeID() -> usize;
        fn CFDictionaryGetTypeID() -> usize;
        fn CFDictionaryGetValue(dictionary: CfType, key: CfType) -> CfType;
        fn CFGetTypeID(object: CfType) -> usize;
        fn CFRelease(object: CfType);
    }

    // SAFETY: as above.
    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOBSDNameMatching(main_port: u32, options: u32, name: *const c_char) -> CfType;
        fn IOServiceGetMatchingService(main_port: u32, matching: CfType) -> u32;
        fn IORegistryEntrySearchCFProperty(
            entry: u32,
            plane: *const c_char,
            key: CfType,
            allocator: CfType,
            options: u32,
        ) -> CfType;
        fn IOObjectRelease(object: u32) -> i32;
    }

    pub(super) fn of(path: &Path) -> Result<Volume, Error> {
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| Error::new(ErrorKind::InvalidInput, "the path holds a zero byte"))?;
        let mut fs = MaybeUninit::<libc::statfs>::uninit();

        // SAFETY: `path` is a C string, and statfs fills one `statfs` struct in `fs` when it succeeds.
        if unsafe { libc::statfs(path.as_ptr(), fs.as_mut_ptr()) } != 0 {
            return Err(Error::last_os_error());
        }

        // SAFETY: statfs succeeded, so it filled every field.
        let fs = unsafe { fs.assume_init() };
        let mount = name(&fs.f_mntonname).ok_or_else(|| Error::other("statfs gave no mount point"))?;
        let device = name(&fs.f_mntfromname).unwrap_or_default();
        let drive = device
            .to_bytes()
            .strip_prefix(b"/dev/")
            .and_then(drive_model)
            .unwrap_or_default();
        Ok(Volume {
            uuid: volume_uuid(&mount)?,
            drive,
            chip: sysctl(c"machdep.cpu.brand_string").unwrap_or_default(),
            os_build: sysctl(c"kern.osversion").unwrap_or_default(),
        })
    }

    /// The C string at the start of `field`, a name that statfs fills.
    fn name(field: &[c_char]) -> Option<CString> {
        let bytes: Vec<u8> = field.iter().map(|&c| c as u8).collect();
        CStr::from_bytes_until_nul(&bytes).ok().map(CStr::to_owned)
    }

    /// What getattrlist writes for the volume's UUID alone, its length and then the UUID's 16 bytes.
    #[repr(C)]
    struct UuidReply {
        length: u32,
        uuid: [u8; 16],
    }

    /// The UUID of the volume mounted at `mount`, in its usual form of 36 upper-case characters.
    fn volume_uuid(mount: &CStr) -> Result<String, Error> {
        let mut list = libc::attrlist {
            bitmapcount: libc::ATTR_BIT_MAP_COUNT,
            reserved: 0,
            commonattr: 0,
            volattr: libc::ATTR_VOL_INFO | libc::ATTR_VOL_UUID,
            dirattr: 0,
            fileattr: 0,
            forkattr: 0,
        };
        let mut reply = UuidReply {
            length: 0,
            uuid: [0; 16],
        };
        // SAFETY: `mount` is a C string, `list` asks for the volume's UUID alone, and getattrlist writes at most
        // `size_of::<UuidReply>()` bytes into `reply`, which is that size.
        let rc = unsafe {
            libc::getattrlist(
                mount.as_ptr(),
                (&raw mut list).cast(),
                (&raw mut reply).cast(),
                size_of::<UuidReply>(),
                0,
            )
        };

        if rc != 0 {
            return Err(Error::last_os_error());
        }

        if (reply.length as usize) < size_of::<UuidReply>() || reply.uuid == [0; 16] {
            return Err(Error::new(ErrorKind::Unsupported, "the volume has no UUID"));
        }

        let hex: Vec<String> = reply.uuid.iter().map(|b| format!("{b:02X}")).collect();
        Ok([&hex[..4], &hex[4..6], &hex[6..8], &hex[8..10], &hex[10..]]
            .map(|part| part.concat())
            .join("-"))
    }

    /// The text that the sysctl `name` holds, eg. "Apple M3 Pro" for machdep.cpu.brand_string.
    fn sysctl(name: &CStr) -> Option<String> {
        let mut len = 0usize;

        // SAFETY: `name` is a C string, and with no buffer sysctlbyname only writes the value's length into `len`.
        if unsafe { libc::sysctlbyname(name.as_ptr(), null_mut(), &mut len, null_mut(), 0) } != 0 || len == 0 {
            return None;
        }

        let mut buffer = vec![0u8; len];

        // SAFETY: `buffer` holds `len` bytes, and sysctlbyname writes at most that many and sets `len` to the count.
        if unsafe { libc::sysctlbyname(name.as_ptr(), buffer.as_mut_ptr().cast(), &mut len, null_mut(), 0) } != 0 {
            return None;
        }

        buffer.truncate(len);
        buffer.push(0);
        let text = CStr::from_bytes_until_nul(&buffer).ok()?.to_str().ok()?.trim();
        (!text.is_empty()).then(|| text.to_string())
    }

    /// A Core Foundation object that this code owns, released on drop.
    struct Owned(CfType);

    impl Owned {
        fn new(object: CfType) -> Option<Owned> {
            (!object.is_null()).then_some(Owned(object))
        }

        fn string(text: &CStr) -> Option<Owned> {
            // SAFETY: `text` is a C string, and CFStringCreateWithCString returns a new string or null.
            Owned::new(unsafe { CFStringCreateWithCString(null(), text.as_ptr(), UTF8) })
        }
    }

    impl Drop for Owned {
        fn drop(&mut self) {
            // SAFETY: the object is live and this code holds one reference to it, which this gives back.
            unsafe { CFRelease(self.0) };
        }
    }

    /// The model of the drive under the disk named `disk`, eg. "disk3s1". It is the "Product Name" in the "Device
    /// Characteristics" of the disk's nearest parent in the IO registry that has them, eg. the drive's controller.
    fn drive_model(disk: &[u8]) -> Option<String> {
        let disk = CString::new(disk).ok()?;
        let key = Owned::string(c"Device Characteristics")?;
        let product = Owned::string(c"Product Name")?;
        // SAFETY: port 0 is the default main port. IOBSDNameMatching returns a new dictionary or null, and
        // IOServiceGetMatchingService takes that dictionary over even when it fails, and returns a service, or 0.
        let service = unsafe {
            let matching = IOBSDNameMatching(0, 0, disk.as_ptr());
            if matching.is_null() {
                return None;
            }
            IOServiceGetMatchingService(0, matching)
        };

        if service == 0 {
            return None;
        }

        // SAFETY: `service` is a live registry entry, the plane and the key are valid, and the search returns a new
        // object or null.
        let found = unsafe {
            IORegistryEntrySearchCFProperty(service, c"IOService".as_ptr(), key.0, null(), RECURSIVELY | PARENTS)
        };
        // SAFETY: IOServiceGetMatchingService gave this code a reference to `service`, which this gives back.
        unsafe { IOObjectRelease(service) };
        let found = Owned::new(found)?;

        // SAFETY: `found` is a live object.
        if unsafe { CFGetTypeID(found.0) != CFDictionaryGetTypeID() } {
            return None;
        }

        // SAFETY: `found` is a live dictionary, and the value it holds for a key lives as long as it does, i.e. through
        // the copy below.
        unsafe { text_of(CFDictionaryGetValue(found.0, product.0)) }
    }

    /// The text of `value`, if it is a string.
    ///
    /// # Safety
    ///
    /// `value` must be null or a live Core Foundation object.
    unsafe fn text_of(value: CfType) -> Option<String> {
        // SAFETY: the caller promises that `value` is null or live, and it isn't null here.
        if value.is_null() || unsafe { CFGetTypeID(value) != CFStringGetTypeID() } {
            return None;
        }

        let mut buffer = [0u8; 256];
        // SAFETY: `value` is a live string, and CFStringGetCString writes at most the buffer's 256 bytes, ending with a
        // zero byte, or fails.
        let copied = unsafe { CFStringGetCString(value, buffer.as_mut_ptr().cast(), buffer.len() as isize, UTF8) };

        if copied == 0 {
            return None;
        }

        let text = CStr::from_bytes_until_nul(&buffer).ok()?.to_str().ok()?.trim();
        (!text.is_empty()).then(|| text.to_string())
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use std::io::{Error, ErrorKind};
    use std::path::Path;

    use super::Volume;

    pub(super) fn of(_path: &Path) -> Result<Volume, Error> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "telling disk volumes apart is only written for macOS",
        ))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/scheduler/volume.rs"]
mod tests;
