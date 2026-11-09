# Data Quality & Validation Report

- **Total Records Evaluated:** 100,006
- **Clean Valid Records:** 100,000
- **Quarantined Invalid Records:** 6
- **Quarantine Location:** `data/errors\invalid_interactions.csv`

## Validation Rules Summary

| Rule Checked | Criteria | Status |
| :--- | :--- | :--- |
| Nulls / Missing Values | Critical fields not null | ENFORCED |
| Rating Domain | 1.0 <= rating <= 5.0 | ENFORCED |
| Timestamp Range | Epoch > 0 & < 2030 | ENFORCED |
| Duplicate Rows | Unique (user, item, timestamp) | ENFORCED |
| Referential Integrity | User & Item must exist in DB | ENFORCED |
