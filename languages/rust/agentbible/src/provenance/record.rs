use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::SPEC_VERSION;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub check_name: String,
    pub passed: bool,
    pub rtol: f64,
    pub atol: f64,
    pub norm_used: String,
    pub error_message: Option<String>,
}

impl CheckResult {
    pub fn success(check_name: &str, rtol: f64, atol: f64, norm_used: &str) -> Self {
        Self {
            check_name: check_name.to_string(),
            passed: true,
            rtol,
            atol,
            norm_used: norm_used.to_string(),
            error_message: None,
        }
    }

    pub fn failure(check_name: &str, rtol: f64, atol: f64, norm_used: &str, message: &str) -> Self {
        Self {
            check_name: check_name.to_string(),
            passed: false,
            rtol,
            atol,
            norm_used: norm_used.to_string(),
            error_message: Some(message.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Metadata {
    pub git_branch: Option<String>,
    pub git_dirty: Option<bool>,
    pub hostname: Option<String>,
    pub platform: Option<String>,
    pub cpu_model: Option<String>,
    pub memory_gb: Option<f64>,
    pub gpu_info: Option<String>,
    pub slurm_job_id: Option<String>,
    pub slurm_nodelist: Option<String>,
    pub mpi_rank: Option<i64>,
    pub mpi_size: Option<i64>,
    pub random_seed_numpy: Option<i64>,
    pub random_seed_python: Option<i64>,
    pub packages: HashMap<String, String>,
    pub pip_freeze: String,
    pub quantum_backend: Option<String>,
    pub quantum_shots: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub spec_version: String,
    pub language: String,
    pub timestamp: String,
    pub git_sha: String,
    pub checks_passed: Vec<CheckResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

impl ProvenanceRecord {
    pub fn new(checks_passed: Vec<CheckResult>) -> Self {
        Self {
            spec_version: SPEC_VERSION.to_string(),
            language: "rust".to_string(),
            timestamp: Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            git_sha: env!("AGENTBIBLE_GIT_SHA").to_string(),
            checks_passed,
            metadata: None,
        }
    }

    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).expect("provenance serialization should succeed")
    }
}

pub fn emit_failure_and_panic(
    check_name: &str,
    rtol: f64,
    atol: f64,
    norm_used: &str,
    message: &str,
) -> ! {
    let record = ProvenanceRecord::new(vec![CheckResult::failure(
        check_name, rtol, atol, norm_used, message,
    )]);
    eprintln!("{}", record.to_json_string());
    panic!("{message}");
}
