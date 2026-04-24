use std::fs;
use std::path::PathBuf;

use agentbible::provenance::{CheckResult, ProvenanceRecord};
use serde_json::Value;

fn schema_path() -> PathBuf {
    PathBuf::from("../../../schema/provenance_v1.json")
}

#[test]
fn provenance_roundtrip_matches_schema_contract() {
    let schema: Value =
        serde_json::from_str(&fs::read_to_string(schema_path()).expect("schema")).expect("json");
    let record = ProvenanceRecord::new(vec![
        CheckResult::success("finite_array", 0.0, 0.0, "n/a"),
        CheckResult::success("unitary", 1e-10, 1e-12, "frobenius"),
    ]);
    let payload: Value = serde_json::from_str(&record.to_json_string()).expect("payload");
    let required = schema["required"].as_array().expect("required");
    for field in required {
        let name = field.as_str().expect("field");
        assert!(payload.get(name).is_some(), "missing field {name}");
    }
    assert_eq!(payload["spec_version"], "1.0");
    assert_eq!(payload["language"], "rust");
}
