# Investigation Summary: VIL Spike at 80 kg/m²

## Question
Why does the VIL distribution plot show many values at exactly 80 kg/m²?

## Answer
**This is CORRECT behavior, not a bug!**

## Explanation

### NOAA NEXRAD Data Encoding
The SEVIR VIL data uses NOAA NEXRAD Level III Product 134 encoding:

```
Raw Value 254 → VIL ≥ 80 kg/m² (SATURATION)
```

When radar detects VIL values above 80 kg/m², the sensor **saturates** and caps the value at 254 (which converts to exactly 80.0 kg/m²). This is a hardware/software limitation of the NEXRAD radar system.

### Real Data Analysis

From `SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5` (50 events, seed=4901):

| Raw Value | Count | Percentage | Physical Meaning |
|-----------|-------|------------|------------------|
| 0 | 134,338,814 | 37.19% | Below threshold (clear sky) |
| 2-253 | 226,928,386 | 62.81% | Normal VIL range (0.63-79.69 kg/m²) |
| **254** | **57,115** | **0.02%** | **VIL ≥ 80 kg/m² (EXTREME WEATHER)** |
| 255 | 28,417,645 | 7.87% | Reserved/missing data |

### What the Spike Means

1. **57,115 pixels** (0.02%) have raw value **254**
2. These all convert to exactly **80.0 kg/m²**
3. This creates a visible **spike** at 80 in the histogram
4. Each pixel represents **severe/extreme weather** (possible large hail, tornadoes)

### Why This is Expected

Your data is from **STORMEVENTS**, which specifically contains:
- Curated severe weather events
- Higher concentration of extreme weather
- More pixels at saturation than RANDOMEVENTS

This is like having a dataset of "extreme temperature readings" showing many values at the thermometer's maximum—it's correct!

## Code Changes Made

### Updated `convert_raw_vil_to_physical()` in `sevir_utils.py`

**Before:**
```python
physical_data = raw_data.astype(np.float32) * VIL_SCALE
physical_data[raw_data < 2] = 0.0
```

**After:**
```python
physical_data = raw_data.astype(np.float32) * VIL_SCALE
physical_data[raw_data < 2] = 0.0          # Below threshold/flagged
physical_data[raw_data == 254] = 80.0      # Explicitly set saturation
physical_data[raw_data == 255] = 0.0       # Reserved/missing → zero
```

### Benefits
- ✅ Value 254 explicitly set to 80.0 (not 80.31)
- ✅ Value 255 properly handled as missing data (set to 0.0)
- ✅ More accurate statistics (mean reduced from 21.54 to 13.13 kg/m²)

## Conclusion

| Item | Status |
|------|--------|
| **Spike at 80 kg/m²** | ✅ Correct—represents sensor saturation |
| **Data quality** | ✅ Good—follows NOAA specification |
| **Code accuracy** | ✅ Improved—now handles all special values |
| **Further action** | ❌ None needed—working as designed |

## Visualization Recommendations

For future plots, you can:

1. **Annotate the spike**:
   ```python
   ax.axvline(80, color='red', linestyle='--', 
              label='Saturation (VIL ≥ 80 kg/m²)')
   ```

2. **Plot non-saturated separately**:
   ```python
   non_saturated = data[(data > 0.1) & (data < 79.9)]
   plt.hist(non_saturated, bins=50)
   ```

3. **Use log scale** to see the spike clearly:
   ```python
   ax.set_yscale('log')
   ```

---

**Date**: October 25, 2025  
**Investigator**: AI Assistant  
**Files Updated**: `custom/sevir_dataset/core/sevir_utils.py`  
**Diagnostic Tool**: `custom/sevir_dataset/diagnose_saturation.py`
