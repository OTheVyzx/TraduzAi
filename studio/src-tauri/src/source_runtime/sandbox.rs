#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SandboxLimits {
    pub(crate) max_processes: u32,
    pub(crate) process_memory_bytes: usize,
    pub(crate) job_memory_bytes: usize,
    pub(crate) cpu_rate: u32,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            max_processes: 9,
            process_memory_bytes: 512 * 1024 * 1024,
            job_memory_bytes: 768 * 1024 * 1024,
            cpu_rate: 5_000,
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::SandboxLimits;
    use std::ffi::c_void;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectCpuRateControlInformation,
        JobObjectExtendedLimitInformation, SetInformationJobObject,
        JOBOBJECT_CPU_RATE_CONTROL_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_CPU_RATE_CONTROL_ENABLE, JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION,
        JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    };

    pub(crate) struct ProcessSandbox {
        job: HANDLE,
    }

    // The HANDLE is owned exclusively and closed in Drop.
    unsafe impl Send for ProcessSandbox {}

    impl ProcessSandbox {
        pub(crate) fn attach(
            child: &tokio::process::Child,
            limits: SandboxLimits,
        ) -> Result<Self, String> {
            let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if job.is_null() {
                return Err(format!(
                    "CreateJobObjectW falhou: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let sandbox = Self { job };

            let mut information = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            information.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
                | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                | JOB_OBJECT_LIMIT_PROCESS_MEMORY
                | JOB_OBJECT_LIMIT_JOB_MEMORY;
            information.BasicLimitInformation.ActiveProcessLimit = limits.max_processes;
            information.ProcessMemoryLimit = limits.process_memory_bytes;
            information.JobMemoryLimit = limits.job_memory_bytes;
            let configured = unsafe {
                SetInformationJobObject(
                    sandbox.job,
                    JobObjectExtendedLimitInformation,
                    (&information as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast::<c_void>(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if configured == 0 {
                return Err(format!(
                    "SetInformationJobObject falhou: {}",
                    std::io::Error::last_os_error()
                ));
            }

            let mut cpu = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION::default();
            cpu.ControlFlags =
                JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP;
            cpu.Anonymous.CpuRate = limits.cpu_rate;
            let cpu_configured = unsafe {
                SetInformationJobObject(
                    sandbox.job,
                    JobObjectCpuRateControlInformation,
                    (&cpu as *const JOBOBJECT_CPU_RATE_CONTROL_INFORMATION).cast::<c_void>(),
                    std::mem::size_of::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>() as u32,
                )
            };
            if cpu_configured == 0 {
                return Err(format!(
                    "Limite de CPU do Job Object falhou: {}",
                    std::io::Error::last_os_error()
                ));
            }

            let process = child
                .raw_handle()
                .ok_or("Processo do runtime não expôs um handle")?
                as HANDLE;
            if unsafe { AssignProcessToJobObject(sandbox.job, process) } == 0 {
                return Err(format!(
                    "AssignProcessToJobObject falhou: {}",
                    std::io::Error::last_os_error()
                ));
            }
            Ok(sandbox)
        }
    }

    impl Drop for ProcessSandbox {
        fn drop(&mut self) {
            if !self.job.is_null() {
                unsafe { CloseHandle(self.job) };
            }
        }
    }
}

#[cfg(not(windows))]
mod platform {
    use super::SandboxLimits;

    pub(crate) struct ProcessSandbox;

    impl ProcessSandbox {
        pub(crate) fn attach(
            _child: &tokio::process::Child,
            _limits: SandboxLimits,
        ) -> Result<Self, String> {
            Err("Sandbox do runtime ainda é suportada apenas no Windows".to_string())
        }
    }
}

pub(crate) use platform::ProcessSandbox;
