# VIL Saturation Analysis

## Summary

The spike at **80 kg/m²** in the VIL distribution plot is **CORRECT** and **EXPECTED** behavior, not an error.

## Why This Happens

### NOAA NEXRAD Specification

According to the NOAA NEXRAD Level III Product 134 (VIL) specification:

| Raw Value (uint8) | Meaning | Physical Value |
|-------------------|---------|----------------|
| 0 | Below threshold | 0.0 kg/m² |
| 1 | Flagged data | 0.0 kg/m² |
| 2-253 | Linear mapping | `raw × (80.0/254.0)` kg/m² |
| **254** | **VIL ≥ 80 kg/m²** | **80.0 kg/m² (saturated)** |
| 255 | Reserved/missing | 0.0 kg/m² |

### Saturation Explained

- **Data level 254** is a **saturation/capping value**
- It represents **ALL** VIL values greater than or equal to 80 kg/m²
- The sensor cannot distinguish between 80, 90, 100, or 150 kg/m²—they all become 254
- This creates a spike in the histogram at exactly 80.0 kg/m²

## Real Data Analysis (SEVIR_VIL_STORMEVENTS_2019)

From diagnostic analysis with seed=4901, 50 events:

```
Special Value Counts (from 361M pixels total):
  Value 0 (below threshold):   134,338,814 (37.19%)  ← Clear sky
  Value 1 (flagged):           0 (0.00%)
  Value 254 (VIL >= 80 kg/m²): 57,115 (0.02%)      ← SATURATION SPIKE
  Value 255 (reserved):        28,417,645 (7.87%)   ← Missing data
```

### Key Findings

1. **57,115 pixels (0.02%)** have raw value 254
2. These all convert to exactly **80.0 kg/m²**
3. This represents **severe weather** with extremely high VIL
4. The percentage is small but visible in histograms

## Physical Interpretation

### Weather Severity Scale (VIL in kg/m²)

| Range | Classification | Significance |
|-------|----------------|--------------|
| 0-20 | Light | Light precipitation, no severe weather |
| 20-45 | Moderate | Moderate precipitation |
| 45-65 | Heavy | Heavy precipitation, possible hail |
| 65-80 | Severe | Very severe storms, hail likely |
| **≥80** | **Extreme** | **Extreme storms, large hail, tornadoes** |

### Why STORMEVENTS Shows More Saturation

The SEVIR dataset has two types:
- **RANDOMEVENTS**: Random weather samples (diverse, less saturation)
- **STORMEVENTS**: Curated severe weather events (**more saturation expected**)

Your data is from **STORMEVENTS**, which is specifically selected for severe weather, so seeing saturation at 80 kg/m² is **completely normal and correct**.

## Conclusion

✅ **The spike at 80 kg/m² is correct**  
✅ **It indicates severe weather in the dataset**  
✅ **This is expected for STORMEVENTS data**  
✅ **No code changes needed—working as designed**

## Technical Notes

### Reserved Value 255

- Found in **7.87%** of pixels in the sample
- Represents missing/invalid data
- Now properly handled by setting to 0.0 kg/m²
- Should be filtered out in analysis (similar to value 0)

### Visualization Recommendations

When plotting distributions:
1. Consider using log scale on y-axis to see the spike clearly
2. Add annotation at 80 kg/m² explaining saturation
3. Can plot "non-saturated" distribution separately (values < 79.9)
4. Filter out zeros and reserved values (255) for cleaner analysis

---

**Generated**: October 25, 2025  
**Data Source**: SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5  
**Sample**: 50 events, seed=4901
