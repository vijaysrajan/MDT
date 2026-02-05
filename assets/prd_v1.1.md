# Metric Decomposition Tree (MDT) - Product Requirements Document

**Version:** 1.1  
**Date:** January 31, 2026  
**Author:** Claude (AI Assistant)  
**Stakeholder:** Vijay

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [Detailed Requirements](#4-detailed-requirements)
5. [Technical Architecture](#5-technical-architecture)
6. [Set Operations Abstraction Layer](#6-set-operations-abstraction-layer) *(NEW in v1.1)*
7. [Configuration Specification](#7-configuration-specification)
8. [Algorithm Specification](#8-algorithm-specification)
9. [Output Specification](#9-output-specification)
10. [Constraints and Validation Rules](#10-constraints-and-validation-rules)
11. [Master Scenario Table](#11-master-scenario-table) *(NEW in v1.1)*
12. [Terminology: Outlier vs DQ Issue vs Fraud vs Anomaly](#12-terminology) *(NEW in v1.1)*
13. [Implementation Plan](#13-implementation-plan)
14. [Epics and User Stories](#14-epics-and-user-stories)
15. [Appendix](#15-appendix)

---

## 1. Executive Summary

The Metric Decomposition Tree (MDT) is a data analysis tool that builds a binary decision tree to decompose a metric of interest while maintaining the MECE (Mutually Exclusive, Collectively Exhaustive) property. The tree identifies which combinations of categorical features (represented as one-hot encoded binary variables) contribute most significantly to a given metric.

The system supports three data input formats:
1. **Raw FACT data** - Transactional CSV files
2. **OLAP Cube** - Pre-aggregated dimensional data in CSV format
3. **Theta Sketches** - Probabilistic data structures for approximate computation at scale

Key capabilities include configurable optimization objectives (maximize, minimize, or balance metric values), support for derived metrics (ratios), frequent itemset mining for non-greedy tree construction, and multiple output formats including LLM-consumable plain English narratives.

---

## 2. Problem Statement

Data analysts and scientists often need to understand how different segments of their data contribute to key business metrics. Traditional approaches either:
- Provide flat aggregations that don't reveal hierarchical relationships
- Use decision trees optimized for prediction rather than decomposition
- Cannot handle approximate computations required for large-scale data

The MDT addresses these gaps by providing a purpose-built tool for metric decomposition that:
- Maintains MECE properties for accurate accounting
- Supports multiple optimization objectives
- Scales via probabilistic data structures
- Produces human-readable explanations

---

## 3. Solution Overview

### 3.1 Core Concept

The MDT builds a binary tree where:
- Each node represents a split on a binary feature (e.g., `city=Mumbai`)
- **Left branch** = Feature is FALSE (absence)
- **Right branch** = Feature is TRUE (presence)
- The tree recursively partitions the data, maintaining MECE at each level

### 3.2 Supported Data Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| Raw FACT | CSV with transactional records, categorical columns, and numeric metrics | Full flexibility, exact computation |
| OLAP Cube | CSV with pre-aggregated data by dimension combinations | Faster computation on pre-computed aggregates |
| Theta Sketch | Probabilistic data structures with set operations | Large-scale approximate computation |

### 3.3 Key Features

- **Configurable optimization**: Maximize, minimize, or find closest-to-middle splits
- **Derived metrics**: Support for ratio metrics (e.g., `revenue / count`)
- **Frequent itemset integration**: Non-greedy tree construction using itemsets
- **Multiple outputs**: In-memory object, JSON, visualization, plain English narrative
- **Pre-filtering**: Apply conditions before tree construction

### 3.4 Itemset Discovery Approach by Configuration

| Data Format | `absence` | `opt_level` | Approach |
|-------------|-----------|-------------|----------|
| Raw FACT | False | 1 | Greedy single feature selection |
| Raw FACT | False | >1 | Traditional FIS (Apriori/FP-Growth) |
| Raw FACT | True | 1 | Greedy single feature selection |
| Raw FACT | True | >1 | ❌ Not supported |
| OLAP Cube | False | 1 | Greedy single feature selection |
| OLAP Cube | False | >1 | Traditional FIS |
| OLAP Cube | True | 1 | Greedy single feature selection |
| OLAP Cube | True | >1 | ❌ Not supported |
| Theta Sketch | False | 1 | Greedy single feature selection |
| Theta Sketch | False | >1 | Traditional FIS |
| Theta Sketch | True | 1 | Greedy single feature selection |
| Theta Sketch | True | >1 | ✅ Union Coverage + De Morgan's Law |

**Why Union Coverage for Absence Mode?**

Traditional FIS fails for negated features because support is inverted (`city≠Mumbai` has ~95% support). This causes combinatorial explosion with no effective pruning. The union coverage method exploits De Morgan's law:

```
count(A≠v1 AND B≠v2) = Total - count(A=v1 OR B=v2)
```

Finding rare absence combinations becomes equivalent to finding high-coverage presence unions, which can be done greedily in O(n × k) time using Theta Sketch union operations.

---

## 4. Detailed Requirements

### 4.1 Functional Requirements

#### FR-1: Data Input
- FR-1.1: System shall read Raw FACT data from CSV files into pandas DataFrame
- FR-1.2: System shall read OLAP Cube data from CSV files into pandas DataFrame
- FR-1.3: System shall accept Theta Sketch objects as input
- FR-1.4: System shall automatically detect and convert categorical columns to one-hot encoded binary features

#### FR-2: Preprocessing
- FR-2.1: System shall drop dimensions with only 1 unique value
- FR-2.2: System shall drop dimensions where cardinality equals number of rows
- FR-2.3: System shall apply pre_filter conditions before tree construction
- FR-2.4: System shall validate that all metrics referenced in configuration exist in data

#### FR-3: Tree Construction
- FR-3.1: System shall build a binary tree up to configured max_depth
- FR-3.2: System shall select splits based on configured optimization objective (max/min/closest-to-middle)
- FR-3.3: System shall support both presence (feature=TRUE) and absence (feature=FALSE) based selection
- FR-3.4: System shall recurse on both TRUE and FALSE branches with appropriate filters
- FR-3.5: System shall terminate when max_depth reached or no valid splits remain

#### FR-4: Frequent Itemset Integration
- FR-4.1: When opt_level > 1 and absence=False, system shall use traditional frequent itemset mining
- FR-4.2: When opt_level > 1 and absence=True, system shall use union coverage method with De Morgan's law (Theta Sketch only)
- FR-4.3: System shall compute effective_opt_level = max(1, opt_level - depth)
- FR-4.4: System shall order itemset features by metric value (descending for max, ascending for min)
- FR-4.5: For traditional FIS, system shall start min_support at 80% and halve until 0.01% floor if no results
- FR-4.6: System shall support configurable max retry attempts for min_support reduction
- FR-4.7: For union coverage method, system shall use greedy selection with O(n × k) complexity

#### FR-5: Metric Computation
- FR-5.1: System shall compute SUM for additive metrics
- FR-5.2: System shall compute derived metrics as SUM(numerator) / SUM(denominator)
- FR-5.3: For Theta Sketches, system shall approximate SUM using bin midpoints × estimated counts
- FR-5.4: System shall compute percentage of metric vs root (pct_of_root)
- FR-5.5: System shall compute percentage of metric vs parent (pct_of_parent)

#### FR-6: Output Generation
- FR-6.1: System shall provide in-memory tree object
- FR-6.2: System shall serialize tree to JSON format
- FR-6.3: System shall generate text-based tree visualization
- FR-6.4: System shall generate graphviz-compatible visualization
- FR-6.5: System shall generate plain English narrative for each root-to-leaf path

### 4.2 Non-Functional Requirements

#### NFR-1: Performance
- NFR-1.1: Theta Sketch operations should complete in O(k) time regardless of data size
- NFR-1.2: System should handle datasets with millions of records (via OLAP or Sketch)

#### NFR-2: Usability
- NFR-2.1: Plain English output should be consumable by LLMs for reasoning
- NFR-2.2: Configuration should be intuitive with sensible defaults

#### NFR-3: Maintainability
- NFR-3.1: Three data format implementations should share common interface patterns
- NFR-3.2: Code should be modular and testable

---

## 5. Technical Architecture

### 5.1 Class Structure

```
          ┌───────────────────────────────────────────┐
          │                 MDTConfig                 │
          │  (Configuration dataclass holding params) │
          └─────────────────┬─────────────────────────┘
                            │
                            ▼
          ┌───────────────────────────────────────────┐
          │             BaseTreeBuilder               │
          │   (Abstract base with common tree logic)  │
          ├───────────────────────────────────────────┤
          │  - build_tree()                           │
          │  - _select_best_feature()                 │
          │  - _build_spine_from_itemset()            │
          │  - _generate_outputs()                    │
          └─────┬───────────────┬───────────────┬─────┘
                │               │               │
                ▼               ▼               ▼
       ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
       │   RawFact    │ │  OLAPCube    │ │ ThetaSketch  │
       │ TreeBuilder  │ │ TreeBuilder  │ │ TreeBuilder  │
       ├──────────────┤ ├──────────────┤ ├──────────────┤
       │ _load_       │ │ _load_       │ │ _load_       │
       │  data()      │ │  data()      │ │  sketches()  │
       │ _compute_    │ │ _compute_    │ │ _compute_    │
       │  metric()    │ │  metric()    │ │  metric()    │
       │ _apply_      │ │ _apply_      │ │ _apply_      │
       │  filter()    │ │  filter()    │ │  filter()    │
       │ _get_        │ │ _get_        │ │ _get_        │
       │  features()  │ │  features()  │ │  features()  │
       └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
              │                │                │
              └────────────────┴────────────────┘
                               │
                               ▼
          ┌───────────────────────────────────────────┐
          │                 MDTNode                   │
          │          (Tree node dataclass)            │
          ├───────────────────────────────────────────┤
          │  - feature: str                           │
          │  - operator: str ("=" or "!=")            │
          │  - value: Any                             │
          │  - metric_value: float                    │
          │  - count: int                             │
          │  - pct_of_root: float                     │
          │  - pct_of_parent: float                   │
          │  - true_child: Optional[MDTNode]          │
          │  - false_child: Optional[MDTNode]         │
          └─────────────────┬─────────────────────────┘
                            │
                            ▼
          ┌───────────────────────────────────────────┐
          │            MDTOutputGenerator             │
          ├───────────────────────────────────────────┤
          │  - to_json()                              │
          │  - to_text_tree()                         │
          │  - to_graphviz()                          │
          │  - to_plain_english()                     │
          └───────────────────────────────────────────┘
```

### 5.2 Theta Sketch Architecture

**Sketches stored (separate, independent):**

```
Dimension sketches (one per dimension=value):
  - city=Mumbai_sketch     (IDs where city=Mumbai)
  - city=Delhi_sketch      (IDs where city=Delhi)
  - gender=Male_sketch     (IDs where gender=Male)
  - ...

Metric bin sketches (global, one per bin):
  - rev_bin_1_sketch       (IDs where revenue in bin 1 range)
  - rev_bin_2_sketch       (IDs where revenue in bin 2 range)
  - ...
  - rev_bin_10_sketch
  - spend_bin_1_sketch
  - ...
```

**At compute time**, intersections are performed dynamically:

```python
# COUNT for city=Mumbai && gender=Female
filter_sketch = intersect(city_mumbai_sketch, gender_female_sketch)
count = filter_sketch.get_estimate()

# SUM(revenue) for city=Mumbai && gender=Female
approx_revenue = 0
for i in range(1, num_bins + 1):
    bin_intersection = intersect(filter_sketch, rev_bin_i_sketch)
    count_in_bin = bin_intersection.get_estimate()
    approx_revenue += count_in_bin * bin_i_midpoint
```

---

## 6. Set Operations Abstraction Layer

### 6.1 Overview: Why Multiple Data Structures?

MDT performs set operations (intersection, union, cardinality) at every node to compute metrics. The choice of underlying data structure has profound implications for accuracy, memory usage, and performance.

**v1.1 introduces an abstraction layer** to support multiple set operation providers, enabling users to select the appropriate data structure based on their use case.

| Aspect | Theta Sketches | Roaring Bitmaps |
|--------|----------------|-----------------|
| **Cardinality** | Approximate (±2-3%) | Exact |
| **Memory** | Fixed size (~16KB) | Proportional to cardinality |
| **Speed at Scale** | O(1) merge | O(n) merge |
| **Best For** | Billions of records, streaming | Millions of records, audit trails |
| **Set Ops** | Union, Intersection, Difference | Union, Intersection, Difference, XOR |
| **Member Query** | No ("Was user X in set?") | Yes |

**Key Insight:** Neither is universally better. The right choice depends on the **use case** and **required precision**.

### 6.2 SetOperationsProvider Interface

To future-proof MDT and allow pluggable backends, we define a `SetOperationsProvider` abstraction:

```
            ┌──────────────────────────────────┐
            │       MDT Tree Builder           │
            └────────────────┬─────────────────┘
                             │ uses
                             ▼
   ┌──────────────────────────────────────────────────┐
   │      SetOperationsProvider (Interface)           │
   │      ────────────────────────────────            │
   │      • create_empty()                            │
   │      • union() / intersect() / difference()      │
   │      • get_estimate()                            │
   │      • is_exact()                                │
   └───────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌───────────────┐ ┌──────────┐ ┌──────────────┐
│  ThetaSketch  │ │ Roaring  │ │    Future    │
│   Provider    │ │  Bitmap  │ │(HyperLogLog, │
│               │ │ Provider │ │  CPC, etc.)  │
│  Approximate  │ │  Exact   │ │              │
│  O(1) merge   │ │  Member  │ │              │
│               │ │  query   │ │              │
└───────────────┘ └──────────┘ └──────────────┘
```

**Interface Definition:**

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional
from enum import Enum

T = TypeVar('T')  # The underlying sketch/bitmap type

class SetOperationsProvider(ABC, Generic[T]):
    """
    Abstract interface for set operations used by MDT.
    Implementations can be Theta Sketches, Roaring Bitmaps, or future structures.
    """
    
    @abstractmethod
    def create_empty(self) -> T:
        """Create an empty set structure."""
        pass
    
    @abstractmethod
    def add(self, structure: T, item: int) -> T:
        """Add an item to the structure."""
        pass
    
    @abstractmethod
    def union(self, structures: List[T]) -> T:
        """Compute union of multiple structures."""
        pass
    
    @abstractmethod
    def intersect(self, a: T, b: T) -> T:
        """Compute intersection of two structures."""
        pass
    
    @abstractmethod
    def difference(self, a: T, b: T) -> T:
        """Compute set difference (a - b)."""
        pass
    
    @abstractmethod
    def get_estimate(self, structure: T) -> float:
        """Get cardinality estimate (exact or approximate)."""
        pass
    
    @abstractmethod
    def is_exact(self) -> bool:
        """Returns True if this provider gives exact counts."""
        pass
    
    @abstractmethod
    def get_error_bound(self, structure: T) -> Optional[float]:
        """Returns error bound (None for exact providers)."""
        pass
    
    @abstractmethod
    def serialize(self, structure: T) -> bytes:
        """Serialize structure for storage."""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> T:
        """Deserialize structure from storage."""
        pass


class ProviderType(Enum):
    THETA_SKETCH = "theta_sketch"
    ROARING_BITMAP = "roaring_bitmap"
    AUTO = "auto"  # System selects based on use case
```

### 6.3 Provider Selection Logic

```python
def select_provider(config: MDTConfig, data_cardinality: int) -> SetOperationsProvider:
    """
    Automatically select the best provider based on config and data characteristics.
    """
    
    # Explicit override
    if config.provider_type == ProviderType.THETA_SKETCH:
        return ThetaSketchProvider()
    if config.provider_type == ProviderType.ROARING_BITMAP:
        return RoaringBitmapProvider()
    
    # Force exact if required
    if config.require_exact:
        return RoaringBitmapProvider()
    
    # Use case based selection
    USE_CASE_PROVIDERS = {
        "data_quality": ProviderType.ROARING_BITMAP,  # DQ needs exact zeros
        "fraud": ProviderType.ROARING_BITMAP,         # Evidence must be exact
        "root_cause": ProviderType.THETA_SKETCH,      # Scale matters more
        "micro_segment": ProviderType.THETA_SKETCH,   # Approximate is fine
        "operational": ProviderType.THETA_SKETCH,     # Logs are massive
        "missing_data": ProviderType.ROARING_BITMAP,  # Null checks need precision
    }
    
    if config.use_case in USE_CASE_PROVIDERS:
        provider_type = USE_CASE_PROVIDERS[config.use_case]
        if provider_type == ProviderType.ROARING_BITMAP:
            return RoaringBitmapProvider()
        return ThetaSketchProvider()
    
    # Heuristic: Use cardinality threshold
    BITMAP_CARDINALITY_THRESHOLD = 100_000_000  # 100M
    
    if data_cardinality > BITMAP_CARDINALITY_THRESHOLD:
        return ThetaSketchProvider()
    
    # Default: If looking for zeros/mins, prefer exact
    if config.max_min_mean == "min":
        return RoaringBitmapProvider()
    
    return ThetaSketchProvider()
```

### 6.4 When to Use Each Provider

| Criterion | Use Theta Sketch | Use Roaring Bitmap |
|-----------|------------------|-------------------|
| Cardinality > 100M | ✅ | ⚠️ Memory concerns |
| Need exact count | ❌ | ✅ |
| Need member query ("Is X in set?") | ❌ | ✅ |
| Streaming/real-time | ✅ | ⚠️ Possible but heavier |
| Finding zeros (DQ) | ⚠️ Approximate zeros ≠ exact | ✅ |
| Legal/compliance evidence | ❌ | ✅ |
| Exploratory analysis | ✅ | ✅ |
| Distributed merge | ✅ (Constant size) | ⚠️ (Size grows) |

### 6.5 Extended MDTConfig for Provider Selection

```python
@dataclass
class MDTConfig:
    # Existing fields
    md: int                          # Max depth
    moi: str                         # Metric of interest
    opt_level: int = 1               # Optimization level
    max_min_mean: str = "max"        # "max", "min", "mid"
    absence: bool = False            # Absence mode
    pre_filter: Optional[str] = None # Pre-filter expression
    
    # NEW: Data structure configuration
    provider_type: ProviderType = ProviderType.AUTO
    require_exact: bool = False      # Force exact counts
    
    # NEW: Use case hint for AUTO mode
    use_case: Optional[str] = None   # "root_cause", "data_quality", "fraud", etc.
```

### 6.6 Validation for Provider Selection

```python
def validate_provider_config(config: MDTConfig, provider: SetOperationsProvider) -> None:
    """
    Validate that the provider is appropriate for the config.
    Raise warnings or errors for mismatches.
    """
    
    # Critical: DQ checks with approximate counts
    if config.use_case == "data_quality" and not provider.is_exact():
        raise ConfigurationError(
            "Data Quality audits require exact counts. "
            "Set provider_type=ROARING_BITMAP or require_exact=True"
        )
    
    # Warning: Fraud detection with approximate
    if config.use_case == "fraud" and not provider.is_exact():
        logger.warning(
            "Fraud detection with approximate counts may miss edge cases. "
            "Consider using Roaring Bitmaps for evidentiary requirements."
        )
    
    # Warning: Large data with bitmaps
    if config.estimated_cardinality > 1_000_000_000 and provider.is_exact():
        logger.warning(
            f"Using exact provider with {config.estimated_cardinality:,} records "
            "may cause memory issues. Consider Theta Sketches."
        )
```

### 6.7 Future Extensibility

The abstraction layer supports future providers:

| Provider | Use Case | Status |
|----------|----------|--------|
| Theta Sketch | Large-scale approximate | ✅ Implemented |
| Roaring Bitmap | Exact counts, member queries | ✅ Implemented |
| HyperLogLog++ | Cardinality-only (lighter than Theta) | 🔮 Future |
| CPC Sketch | Higher accuracy than HLL | 🔮 Future |
| MinHash | Jaccard similarity | 🔮 Future |
| Count-Min Sketch | Frequency estimation | 🔮 Future |
| Bloom Filter | Membership testing (probabilistic) | 🔮 Future |

---

## 7. Configuration Specification

### 7.1 Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `md` | int | Yes | - | Maximum depth of the tree |
| `moi` | str | Yes | - | Metric of interest (name or formula like `"revenue / count"`) |
| `opt_level` | int | No | 1 | Optimization level; >1 enables frequent itemset mining |
| `max_min_mean` | str | No | "max" | Optimization objective: "max", "min", or "closest-to-middle" |
| `absence` | bool | No | False | If True, select features by FALSE metric; if False, by TRUE metric |
| `pre_filter` | str | No | None | Filter expression (e.g., `"city=Mumbai && gender=Male && agegroup!=55-60"`) |
| `use_frequent_itemsets` | bool | No | False | Explicitly enable frequent itemset mining |
| `fis_min_support_start` | float | No | 0.80 | Starting min_support for frequent itemset mining |
| `fis_min_support_floor` | float | No | 0.0001 | Minimum min_support (0.01%) |
| `fis_max_retries` | int | No | 10 | Maximum retry attempts for min_support reduction |

### 7.2 Configuration Validation Rules

1. `md` must be a positive integer
2. `moi` must reference valid metrics (either directly or in formula)
3. `opt_level` must be >= 1
4. `max_min_mean` must be one of: "max", "min", "closest-to-middle"
5. If `opt_level > 1` and `max_min_mean == "closest-to-middle"`, raise error (not supported)
6. If `opt_level > 1` and `absence == True` and data format is not Theta Sketch, raise error
7. If `opt_level > md`, elevate `md` to `opt_level`

### 7.3 Configuration Examples

**Example 1: Simple max decomposition**
```python
config = MDTConfig(
    md=4,
    moi="revenue",
    max_min_mean="max"
)
```

**Example 2: Derived metric with filtering**
```python
config = MDTConfig(
    md=5,
    moi="clicks / impressions",
    max_min_mean="max",
    pre_filter="channel=organic && device!=tablet"
)
```

**Example 3: Frequent itemset-based tree**
```python
config = MDTConfig(
    md=6,
    moi="spend",
    opt_level=3,
    max_min_mean="min",
    use_frequent_itemsets=True
)
```

---

## 8. Algorithm Specification

### 8.1 Main Tree Building Algorithm

```
FUNCTION build_tree(data, depth, opt_level, max_depth, path_filter, parent_metric, root_metric, config):
    
    # Termination conditions
    IF depth >= max_depth:
        RETURN Leaf(compute_metric(data, path_filter), ...)
    
    filtered_data = apply_filter(data, path_filter)
    
    IF no_valid_splits_remain(filtered_data):
        RETURN Leaf(compute_metric(filtered_data), ...)
    
    # Compute effective optimization level
    effective_opt_level = max(1, opt_level - depth)
    
    # Get candidate features and select best
    IF effective_opt_level > 1 AND use_frequent_itemsets:
        
        IF config.absence == False:
            # Traditional FIS mining (works for all data formats)
            itemset = mine_frequent_itemset(filtered_data, size=effective_opt_level)
            ordered_features = sort_itemset_by_metric(itemset, config.max_min_mean)
        ELSE:
            # Union coverage method (Theta Sketch only)
            # Note: Validation ensures we only reach here with Theta Sketch
            selected_features, absence_count = find_k_absence_itemset_via_union(
                sketches, universal_sketch, effective_opt_level, config.max_min_mean
            )
            ordered_features = order_absence_itemset_for_spine(
                selected_features, sketches, universal_sketch, metric_bins, config.max_min_mean
            )
        
        RETURN build_spine_from_itemset(ordered_features, ...)
    
    ELSE:
        best_feature = select_best_feature(filtered_data, config)
        node = create_node(best_feature, ...)
        
        # Recurse on both branches
        node.false_child = build_tree(
            data, depth + 1, opt_level, max_depth,
            path_filter + [(feature, FALSE)], node.metric, root_metric, config
        )
        node.true_child = build_tree(
            data, depth + 1, opt_level, max_depth,
            path_filter + [(feature, TRUE)], node.metric, root_metric, config
        )
        
        RETURN node
```

### 8.2 Feature Selection Algorithm

```
FUNCTION select_best_feature(data, absence, max_min_mean, parent_metric):
    
    candidate_features = get_available_features(data)
    
    FOR each feature IN candidate_features:
        IF absence == False:
            metric[feature] = compute_metric(data WHERE feature == TRUE)
        ELSE:
            metric[feature] = compute_metric(data WHERE feature == FALSE)
    
    IF max_min_mean == "max":
        RETURN feature with maximum metric value
    ELSE IF max_min_mean == "min":
        RETURN feature with minimum metric value
    ELSE IF max_min_mean == "closest-to-middle":
        target = parent_metric / 2
        RETURN feature with metric closest to target
    
    # Tie-breaker: random selection
    IF multiple features have same metric:
        RETURN random choice among tied features
```

### 8.3 Frequent Itemset Spine Building Algorithm

This algorithm builds the TRUE-branch spine from an ordered list of features obtained either via:
- Traditional FIS mining (presence mode, Raw FACT/OLAP/Theta Sketch)
- Union coverage method (absence mode, Theta Sketch only)

```
FUNCTION build_spine_from_itemset(ordered_features, data, depth, opt_level, max_depth, path_filter, root_metric):
    
    # Build the TRUE-branch spine
    spine_nodes = []
    current_filter = path_filter
    
    FOR i, feature IN enumerate(ordered_features):
        node = create_node(feature, ...)
        spine_nodes.append(node)
        
        IF i > 0:
            spine_nodes[i-1].true_child = node
        
        current_filter = current_filter + [(feature, TRUE)]
    
    # Leaf at end of spine
    spine_nodes[-1].true_child = Leaf(...)
    
    # Fill FALSE branches via in-order traversal
    FOR i, node IN enumerate(spine_nodes):
        spine_depth = depth + i
        false_filter = path_filter
        
        # Add TRUE conditions for all ancestors in spine
        FOR j IN range(i):
            false_filter = false_filter + [(ordered_features[j], TRUE)]
        
        # Add FALSE condition for current node
        false_filter = false_filter + [(ordered_features[i], FALSE)]
        
        # Recurse
        node.false_child = build_tree(
            data, spine_depth + 1, opt_level, max_depth,
            false_filter, node.metric, root_metric
        )
    
    RETURN spine_nodes[0]  # Return root of spine
```

### 8.4 Frequent Itemset Mining with Retry (Presence Mode)

This algorithm applies when `absence=False` and `opt_level > 1` for Raw FACT and OLAP data formats.

```
FUNCTION mine_frequent_itemset(data, size, min_support_start, min_support_floor, max_retries):
    
    min_support = min_support_start  # Start at 80%
    retries = 0
    
    WHILE retries < max_retries:
        itemsets = FIS_library.mine(data, min_support=min_support, max_size=size)
        itemsets_of_target_size = filter(itemsets, size=size)
        
        IF itemsets_of_target_size is not empty:
            RETURN best_itemset_by_metric(itemsets_of_target_size)
        
        # Halve min_support and retry
        min_support = min_support / 2
        
        IF min_support < min_support_floor:
            BREAK
        
        retries += 1
    
    # Fallback: return greedy single feature
    RETURN [select_best_single_feature(data)]
```

### 8.5 Absence-Based Itemset Discovery via Union Coverage (Theta Sketch Only)

**Why Traditional FIS Fails for Absence Mode:**

Traditional frequent itemset mining cannot be used for absence-based discovery because negated features have inverted support:
- `city=Mumbai` → 5% support (sparse, mineable)
- `city≠Mumbai` → 95% support (dense)

When mining negated features with any reasonable support threshold, almost every combination is "frequent," causing a combinatorial explosion.

**The Solution: Union Coverage + De Morgan's Law**

Instead of mining rare negated itemsets directly, we exploit the mathematical relationship:

```
count(A≠v1 AND B≠v2 AND C≠v3) = Total - count(A=v1 OR B=v2 OR C=v3)
```

By De Morgan's law:
```
NOT(A=v1) AND NOT(B=v2) AND NOT(C=v3) = NOT(A=v1 OR B=v2 OR C=v3)
```

Therefore:
- **Rare absence combination** = Complement of **high-coverage union of presence features**
- **Common absence combination** = Complement of **low-coverage union of presence features**

**Algorithm:**

```
FUNCTION find_k_absence_itemset_via_union(sketches, universal_sketch, k, max_min_mean):
    """
    For absence=True with opt_level=k (Theta Sketch only):
    
    - If max_min_mean="min": Find k features whose ABSENCE is RAREST
      → Equivalent to: Find k PRESENCE features with MAXIMUM union coverage
      
    - If max_min_mean="max": Find k features whose ABSENCE is MOST COMMON  
      → Equivalent to: Find k PRESENCE features with MINIMUM union coverage
    
    Note: "closest-to-middle" is not supported with opt_level > 1
    """
    
    total_count = universal_sketch.get_estimate()
    candidate_features = list(sketches.keys())
    
    selected_features = []
    remaining_features = set(candidate_features)
    
    FOR i IN range(k):
        best_next_feature = None
        
        IF max_min_mean == "min":
            # We want rare absence → maximize presence union coverage
            best_coverage = 0
            
            FOR feature IN remaining_features:
                test_combination = selected_features + [feature]
                coverage = compute_union_coverage(test_combination, sketches)
                
                IF coverage > best_coverage:
                    best_coverage = coverage
                    best_next_feature = feature
        
        ELSE IF max_min_mean == "max":
            # We want common absence → minimize presence union coverage
            best_coverage = infinity
            
            FOR feature IN remaining_features:
                test_combination = selected_features + [feature]
                coverage = compute_union_coverage(test_combination, sketches)
                
                IF coverage < best_coverage:
                    best_coverage = coverage
                    best_next_feature = feature
        
        IF best_next_feature is not None:
            selected_features.append(best_next_feature)
            remaining_features.remove(best_next_feature)
    
    # Compute final absence count
    union_coverage = compute_union_coverage(selected_features, sketches)
    absence_count = total_count - union_coverage
    
    RETURN selected_features, absence_count


FUNCTION compute_union_coverage(features, sketches):
    """
    Compute count of records having AT LEAST ONE of the specified features.
    Uses Theta Sketch union operation.
    """
    IF features is empty:
        RETURN 0
    
    union_sketch = sketches[features[0]].copy()
    
    FOR feature IN features[1:]:
        union_sketch = union_sketch.union(sketches[feature])
    
    RETURN union_sketch.get_estimate()
```

**Complexity Analysis:**

| Approach | Time Complexity | Space Complexity |
|----------|-----------------|------------------|
| Traditional FIS on negated features | O(2^n) — explosion | O(2^n) |
| Union coverage greedy selection | O(n × k) | O(sketch_size) |

**Ordering Features for Spine Construction:**

After selecting k features via union coverage, we need to order them for the spine. Since we're in absence mode, we compute the **absence metric** for each individual feature:

```
FUNCTION order_absence_itemset_for_spine(selected_features, sketches, universal_sketch, metric_bins, max_min_mean):
    """
    Order the k selected features by their individual absence metric.
    """
    
    feature_metrics = {}
    
    FOR feature IN selected_features:
        # Compute metric for feature=FALSE (absence)
        absence_sketch = universal_sketch.difference(sketches[feature])
        absence_metric = compute_metric_from_sketch(absence_sketch, metric_bins)
        feature_metrics[feature] = absence_metric
    
    IF max_min_mean == "min":
        # Ascending order: smallest absence metric first
        RETURN sorted(selected_features, key=feature_metrics.get, ascending=True)
    ELSE:
        # Descending order: largest absence metric first
        RETURN sorted(selected_features, key=feature_metrics.get, ascending=False)
```

### 8.6 Metric Computation for Theta Sketches

```
FUNCTION compute_metric_from_sketches(filter_sketch, metric_name, bin_sketches, bin_midpoints):
    
    IF metric_name is simple (e.g., "count"):
        RETURN filter_sketch.get_estimate()
    
    IF metric_name is additive (e.g., "revenue"):
        approx_sum = 0
        FOR i IN range(num_bins):
            bin_intersection = intersect(filter_sketch, bin_sketches[metric_name][i])
            count_in_bin = bin_intersection.get_estimate()
            approx_sum += count_in_bin * bin_midpoints[metric_name][i]
        RETURN approx_sum
    
    IF metric_name is derived (e.g., "revenue / count"):
        numerator = compute_metric_from_sketches(filter_sketch, "revenue", ...)
        denominator = compute_metric_from_sketches(filter_sketch, "count", ...)
        RETURN numerator / denominator
```

---

## 9. Output Specification

### 9.1 Node Structure

Each tree node contains:

| Field | Type | Description |
|-------|------|-------------|
| `feature` | str | Feature name (e.g., "city=Mumbai" or "E10") |
| `operator` | str | "=" for presence, "!=" for absence |
| `value` | Any | Feature value |
| `metric_value` | float | Absolute metric value at this node |
| `count` | int | Number of records / estimated count |
| `pct_of_root` | float | `(metric_value / root_metric) * 100` |
| `pct_of_parent` | float | `(metric_value / parent_metric) * 100` |
| `depth` | int | Depth in tree (root = 0) |
| `is_leaf` | bool | True if no children |
| `true_child` | MDTNode | Right child (feature = TRUE) |
| `false_child` | MDTNode | Left child (feature = FALSE) |

### 9.2 JSON Output Format

```json
{
  "config": {
    "md": 4,
    "moi": "revenue",
    "opt_level": 1,
    "max_min_mean": "max",
    "absence": false,
    "pre_filter": null
  },
  "root_metric": 1520000,
  "root_count": 10000,
  "tree": {
    "feature": "city=Mumbai",
    "operator": "=",
    "value": "Mumbai",
    "metric_value": 1520000,
    "count": 10000,
    "pct_of_root": 100.0,
    "pct_of_parent": null,
    "depth": 0,
    "is_leaf": false,
    "false_child": {
      "feature": "gender=Male",
      "metric_value": 740000,
      "pct_of_root": 48.68,
      "pct_of_parent": 48.68,
      ...
    },
    "true_child": {
      "feature": "agegroup=25-34",
      "metric_value": 780000,
      "pct_of_root": 51.32,
      "pct_of_parent": 51.32,
      ...
    }
  }
}
```

### 9.3 Text Tree Visualization

```
[revenue: $1,520,000 | 100.0% of root]
├── city=Mumbai = FALSE
│   [revenue: $740,000 | 48.7% of root | 48.7% of parent]
│   ├── gender=Male = FALSE
│   │   [revenue: $320,000 | 21.1% of root | 43.2% of parent]
│   │   └── (leaf)
│   └── gender=Male = TRUE
│       [revenue: $420,000 | 27.6% of root | 56.8% of parent]
│       └── (leaf)
└── city=Mumbai = TRUE
    [revenue: $780,000 | 51.3% of root | 51.3% of parent]
    ├── agegroup=25-34 = FALSE
    │   [revenue: $350,000 | 23.0% of root | 44.9% of parent]
    │   └── (leaf)
    └── agegroup=25-34 = TRUE
        [revenue: $430,000 | 28.3% of root | 55.1% of parent]
        └── (leaf)
```

### 9.4 Plain English Narrative

Each root-to-leaf path generates one paragraph:

```
Path 1: Starting with all data (Revenue: $1,520,000, 100.0% of total), 
we first examine records where city=Mumbai is FALSE, which accounts for 
$740,000 (48.7% of total, 48.7% of parent). Within this segment, records 
where gender=Male is FALSE contribute $320,000 (21.1% of total, 43.2% of 
parent segment).

Path 2: Starting with all data (Revenue: $1,520,000, 100.0% of total), 
we first examine records where city=Mumbai is FALSE, which accounts for 
$740,000 (48.7% of total, 48.7% of parent). Within this segment, records 
where gender=Male is TRUE contribute $420,000 (27.6% of total, 56.8% of 
parent segment).

Path 3: Starting with all data (Revenue: $1,520,000, 100.0% of total), 
we first examine records where city=Mumbai is TRUE, which accounts for 
$780,000 (51.3% of total, 51.3% of parent). Within this segment, records 
where agegroup=25-34 is FALSE contribute $350,000 (23.0% of total, 44.9% 
of parent segment).

Path 4: Starting with all data (Revenue: $1,520,000, 100.0% of total), 
we first examine records where city=Mumbai is TRUE, which accounts for 
$780,000 (51.3% of total, 51.3% of parent). Within this segment, records 
where agegroup=25-34 is TRUE contribute $430,000 (28.3% of total, 55.1% 
of parent segment).
```

---

## 10. Constraints and Validation Rules

### 10.1 Configuration Constraints

| Constraint | Rule | Error Message |
|------------|------|---------------|
| C1 | `md` >= 1 | "max_depth must be at least 1" |
| C2 | `opt_level` >= 1 | "opt_level must be at least 1" |
| C3 | `max_min_mean` in ["max", "min", "closest-to-middle"] | "Invalid optimization objective" |
| C4 | If `opt_level > 1` and `max_min_mean == "closest-to-middle"` | "Frequent itemsets cannot be used with closest-to-middle" |
| C5 | If `opt_level > 1` and `absence == True` and format != ThetaSketch | "Absence mode with opt_level > 1 only supported for Theta Sketch (uses union coverage method, not traditional FIS)" |
| C6 | If `opt_level > md`, set `md = opt_level` | (auto-correction, no error) |

### 10.2 Data Validation

| Validation | Action |
|------------|--------|
| Dimension with 1 unique value | Drop dimension with warning |
| Dimension with cardinality == row count | Drop dimension with warning |
| Metric referenced in `moi` not found | Raise error |
| `pre_filter` references invalid dimension/value | Raise error |
| Empty dataset after `pre_filter` | Raise error |

### 10.3 Runtime Constraints

| Constraint | Handling |
|------------|----------|
| All candidates have same metric | Random selection |
| No valid splits remain | Create leaf node |
| Frequent itemset mining returns empty | Halve min_support and retry (up to max_retries) |
| Min_support below floor (0.01%) | Fallback to greedy single feature |

### 10.4 Equivalence Documentation

The following configurations produce identical trees:
- `(absence=False, max_min_mean="max")` ≡ `(absence=True, max_min_mean="min")`
- `(absence=False, max_min_mean="min")` ≡ `(absence=True, max_min_mean="max")`

The `absence` parameter only produces different results when `max_min_mean="closest-to-middle"`.

---

## 11. Master Scenario Table

This section maps business scenarios to specific MDT configurations, providing guidance on when to use each combination of parameters.

### 11.1 Configuration Quick Reference

| Config | Selection Basis | Reveals | Business Question |
|--------|-----------------|---------|-------------------|
| `(absence=False, max)` | Feature=TRUE with highest metric | Big Rocks - dominant segments | "Where is revenue concentrated?" |
| `(absence=False, min)` | Feature=TRUE with lowest metric | Anomalies - rare segments | "Which profiles are unusually rare?" |
| `(absence=False, mid)` | Feature=TRUE closest to parent/2 | Balanced decomposition | "How to partition evenly?" |
| `(absence=True, max)` | Feature=FALSE with highest metric | ≡ `(absence=False, min)` | (Equivalent tree) |
| `(absence=True, min)` | Feature=FALSE with lowest metric | ≡ `(absence=False, max)` | (Equivalent tree) |
| `(absence=True, mid)` | Feature=FALSE closest to parent/2 | Absence-based balance | "How do segments differ by what's missing?" |

### 11.2 Scenario-to-Configuration Matrix

| Scenario | Goal | MDT Mode | `max_min_mean` | `opt_level` | `absence` | Provider | Rationale |
|----------|------|----------|----------------|-------------|-----------|----------|-----------|
| **Business Root Cause** | Explain "Why is Revenue/Churn high?" | Standard | `max` | 1 (or 2) | `False` | **Theta Sketch** | Scale matters; ±2% error acceptable |
| **Data Quality Audit** | Find impossible combinations (e.g., Pregnant Males) | Audit | `min` | 2 | `False` | **Roaring Bitmap** | Must find exact zeros; evidence for correction |
| **Micro-Segment Discovery** | Find hidden niches (Age+Location with high ROI) | Deep Dive | `max` | 2 | `False` | **Either** | Theta usually sufficient |
| **Fraud Detection** | Find suspicious rare patterns | Audit | `min` | 2 | `False` | **Roaring Bitmap** | Evidence must be exact for legal/compliance |
| **Operational Outliers** | Find system spikes (Latency > 10s) | Standard | `max` | 1 | `False` | **Theta Sketch** | Log volumes too large for bitmaps |
| **Missing Data Analysis** | Find segments with systematic nulls | Absence | `max` | 1 | `True` | **Roaring Bitmap** | Null counts need precision |
| **Churn Drivers** | Identify characteristics of churned users | Standard | `max` | 1 | `False` | **Theta Sketch** | Approximate segments sufficient |
| **Negative Correlation** | Find mutually exclusive feature pairs | Audit | `min` | 2 | `False` | **Roaring Bitmap** | Need exact intersection = 0 |

### 11.3 Industry Examples

**E-Commerce/Retail:**
- Big rocks `(F, max, revenue)`: "Premium members in metro cities buying electronics = 60% revenue"
- Anomalies `(F, min, count)`: "Only 3 transactions: premium+rural+COD+electronics - potential fraud"
- Low performers `(F, min, revenue)`: "Guest users on mobile from rural = 0.1% - why?"

**Healthcare/Insurance:**
- Cost concentration `(F, max, claim_amount)`: "Chronic disease patients 60+ in urban hospitals = 70% costs"
- Fraud detection `(F, min, count)`: "Only 5 claims: young+rural+high_amount+outpatient"
- Rare diagnosis `(F, min, patient_count)`: "Diabetes+Heart Disease+Age<30 = 3 patients - data quality issue?"

**Financial Services/Banking:**
- Volume `(F, max, txn_amount)`: "Corporate wire transfers >$100K = 80% volume"
- Fraud `(F, min, count)`: "Only 7 txns: new_account+international+large+night - suspicious"
- Risk segmentation `(F, mid, exposure)`: "Balanced risk distribution view"

**Telecommunications:**
- Data usage `(F, max, data_gb)`: "Unlimited urban users streaming video = 75% bandwidth"
- Churn `(F, max, churned_count)`: "Contract-end+billing_issue+support_call = 40% churn"
- Network anomalies `(F, min, call_count)`: "Only 2 calls: rural+new_tower+roaming - network issue"

---

## 12. Terminology: Outlier vs DQ Issue vs Fraud vs Anomaly

MDT can detect various types of unusual patterns. It is critical to use precise terminology when presenting findings to stakeholders.

### 12.1 Definitions

| Term | Definition | Example | Data Validity | MDT Detection |
|------|------------|---------|---------------|---------------|
| **Outlier** | Statistical deviation; significantly different from other observations | Bill Gates in average income calculation | ✅ Valid | `max` → Find highest contributors |
| **Data Quality Issue** | Violates hard constraint of reality or business logic; **impossible** | Pregnant Male, Age = -5, Created > Transaction Date | ❌ Invalid | `min` + Roaring → Find non-zero impossibles |
| **Fraud** | Technically valid but contextually suspicious; **deceptive intent** | Login from NYC, then London in 10 min; New account + $10K purchase | ✅ Valid types, ⚠️ Suspicious behavior | `min` (rare patterns) + Roaring |
| **Anomaly** | Umbrella term for any non-conforming pattern | Sudden traffic drop (could be holiday, crash, or attack) | ❓ Requires investigation | Any config depending on goal |

### 12.2 Relationship Diagram

```
                 ┌─────────────────────────────┐
                 │         ANOMALY             │
                 │    (Umbrella Term)          │
                 └──────────────┬──────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
         ┌──────────┐    ┌──────────┐    ┌─────────┐
         │ OUTLIER  │    │ DQ ISSUE │    │  FRAUD  │
         │          │    │          │    │         │
         │  Valid   │    │ Invalid  │    │ Valid   │
         │  Real    │    │  Error   │    │Malicious│
         └──────────┘    └─────┬────┘    └─────────┘
                               │
                      ┌────────┴────────┐
                      │                 │
                      ▼                 ▼
                 ┌─────────┐       ┌─────────┐
                 │ System  │       │  Human  │
                 │  Error  │       │  Error  │
                 └─────────┘       └─────────┘
```

### 12.3 Finding Classification in Output

MDT can classify findings based on configuration and results:

```python
class FindingType(Enum):
    SEGMENT = "segment"           # Standard decomposition finding
    DATA_ERROR = "data_error"     # DQ issue (impossible combination)
    RISK = "risk"                 # Fraud/suspicious pattern
    OUTLIER = "outlier"           # Statistical extreme

def classify_finding(config: MDTConfig, node: MDTNode) -> FindingType:
    """
    Classify what type of finding this node represents.
    """
    if config.max_min_mean == "min" and config.use_case == "data_quality":
        if node.count > 0:  # Found non-zero impossible combination
            return FindingType.DATA_ERROR
    
    if config.max_min_mean == "min" and config.use_case == "fraud":
        return FindingType.RISK
    
    if config.max_min_mean == "max" and node.pct_of_root > 50:
        return FindingType.SEGMENT  # Dominant segment
    
    if config.max_min_mean == "max" and node.pct_of_root < 1:
        return FindingType.OUTLIER  # Statistical extreme
    
    return FindingType.SEGMENT
```

### 12.4 Plain English Output by Finding Type

When MDT's AI Agent presents findings to stakeholders, it should use appropriate language:

| Finding Type | Narrative Template |
|--------------|-------------------|
| **SEGMENT** | "I found a **Segment** driving {metric}: {path} accounts for {pct}% of total." |
| **DATA_ERROR** | "I found a **Data Error**: {path} has {count} records but should have 0. This combination is logically impossible and indicates a data quality issue." |
| **RISK** | "I found a **Risk Pattern**: {path} is extremely rare ({count} records, {pct}%) and matches suspicious behavior criteria. Recommend investigation." |
| **OUTLIER** | "I found an **Outlier Segment**: {path} represents only {pct}% but shows statistically significant deviation from the norm." |

### 12.5 DQ Violation Detection with Roaring Bitmaps

For Data Quality use cases, MDT can find combinations that **should be zero but aren't**:

```python
def find_dq_violations(
    provider: RoaringBitmapProvider,
    constraint_pairs: List[Tuple[str, str]],  # e.g., [("gender=Male", "condition=Pregnant")]
    dimension_bitmaps: Dict[str, RoaringBitmap]
) -> List[DQViolation]:
    """
    Find data quality violations where mutually exclusive features co-occur.
    """
    violations = []
    
    for feature_a, feature_b in constraint_pairs:
        bitmap_a = dimension_bitmaps.get(feature_a)
        bitmap_b = dimension_bitmaps.get(feature_b)
        
        if bitmap_a is None or bitmap_b is None:
            continue
        
        # Exact intersection
        intersection = provider.intersect(bitmap_a, bitmap_b)
        count = provider.get_estimate(intersection)  # Exact count
        
        if count > 0:
            violations.append(DQViolation(
                feature_a=feature_a,
                feature_b=feature_b,
                count=int(count),
                record_ids=list(intersection)[:100],  # Sample for investigation
                severity="HIGH" if count > 100 else "MEDIUM"
            ))
    
    return violations
```

---

## 13. Implementation Plan

### 13.1 Phase Overview

| Phase | Duration | Focus |
|-------|----------|-------|
| Phase 1 | Week 1-2 | Core infrastructure and Raw FACT implementation |
| Phase 2 | Week 3 | OLAP Cube implementation |
| Phase 3 | Week 4 | Theta Sketch implementation |
| Phase 4 | Week 5 | Frequent Itemset integration |
| Phase 5 | Week 6 | Output generation and testing |
| Phase 6 | Week 7 | Set Operations Abstraction Layer *(NEW in v1.1)* |

### 13.2 Detailed Timeline

#### Phase 1: Core Infrastructure (Week 1-2)

**Week 1:**
- Day 1-2: Set up project structure, define dataclasses (MDTConfig, MDTNode)
- Day 3-4: Implement configuration validation
- Day 5: Implement pre_filter parser and validator

**Week 2:**
- Day 1-2: Implement RawFactTreeBuilder._load_data() and preprocessing
- Day 3-4: Implement greedy feature selection (opt_level=1)
- Day 5: Implement recursive tree building

#### Phase 2: OLAP Cube (Week 3)

- Day 1-2: Implement OLAPCubeTreeBuilder._load_data()
- Day 3-4: Implement metric computation from aggregates
- Day 5: Test equivalence between Raw FACT and OLAP outputs

#### Phase 3: Theta Sketch (Week 4)

- Day 1-2: Define sketch interface and data structures
- Day 3: Implement sketch intersection operations
- Day 4: Implement SUM approximation via binned sketches
- Day 5: Implement ThetaSketchTreeBuilder

#### Phase 4: Frequent Itemset Integration (Week 5)

- Day 1-2: Integrate FIS library with configurable interface
- Day 3: Implement spine building from itemsets
- Day 4: Implement min_support retry logic
- Day 5: Implement negated itemsets for Theta Sketch

#### Phase 5: Output and Testing (Week 6)

- Day 1: Implement JSON serialization
- Day 2: Implement text tree visualization
- Day 3: Implement graphviz output
- Day 4: Implement plain English narrative generation
- Day 5: End-to-end testing and documentation

#### Phase 6: Set Operations Abstraction Layer (Week 7) *(NEW in v1.1)*

- Day 1: Define SetOperationsProvider interface
- Day 2: Implement ThetaSketchProvider
- Day 3: Implement RoaringBitmapProvider
- Day 4: Implement AUTO provider selection and use_case hints
- Day 5: Add finding classification and update output generators

### 13.3 Dependencies

```
  ┌─────────┐          ┌─────────┐          ┌─────────┐
  │ Phase 1 │─────────►│ Phase 2 │          │ Phase 3 │
  │  Core   │          │  OLAP   │          │  Theta  │
  └────┬────┘          └────┬────┘          └────┬────┘
       │                    │                     │
       │                    ▼                     │
       │               ┌─────────┐                │
       └──────────────►│ Phase 4 │◄───────────────┘
                       │   FIS   │
                       └────┬────┘
                            │
                            ▼
                       ┌─────────┐          ┌─────────┐
                       │ Phase 5 │─────────►│ Phase 6 │
                       │ Output  │          │Provider │
                       └─────────┘          └─────────┘
```

### 13.4 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| FIS library integration complexity | Define abstract interface early; allow pluggable implementations |
| Theta Sketch accuracy concerns | Add accuracy metrics to output; document approximation bounds |
| Performance with large trees | Implement early termination; add progress logging |
| Roaring Bitmap memory usage | Add cardinality threshold warnings; recommend Theta for >100M records |
| Provider selection confusion | Implement AUTO mode with clear heuristics; document use cases |

---

## 14. Epics and User Stories

### Epic 1: Data Input and Preprocessing

**E1: As a data analyst, I want to load data from multiple formats so that I can use the tool regardless of my data infrastructure.**

#### User Stories

**US-1.1: Load Raw FACT Data**
> As a data analyst, I want to load transactional data from a CSV file so that I can analyze raw records.

**Acceptance Criteria:**
- System reads CSV file into pandas DataFrame
- System identifies categorical columns automatically
- System identifies numeric metric columns
- System handles missing values appropriately
- System reports data summary (row count, column types)

**US-1.2: Load OLAP Cube Data**
> As a data analyst, I want to load pre-aggregated cube data from a CSV file so that I can analyze without recomputing aggregates.

**Acceptance Criteria:**
- System reads aggregated CSV with dimension and metric columns
- System validates that all dimension combinations are present
- System handles partial cubes gracefully

**US-1.3: Load Theta Sketch Data**
> As a data engineer, I want to provide Theta Sketch objects so that I can analyze large-scale data approximately.

**Acceptance Criteria:**
- System accepts dictionary of dimension sketches
- System accepts dictionary of metric bin sketches
- System validates sketch compatibility
- System reports estimated cardinality

**US-1.4: Preprocess Dimensions**
> As a data analyst, I want the system to automatically clean dimensions so that I don't get meaningless splits.

**Acceptance Criteria:**
- System drops dimensions with single unique value
- System drops dimensions with cardinality equal to row count
- System logs dropped dimensions with reasons
- System converts categorical columns to one-hot encoded features

**US-1.5: Apply Pre-filter**
> As a data analyst, I want to filter data before analysis so that I can focus on specific segments.

**Acceptance Criteria:**
- System parses filter expressions (AND, equality, inequality)
- System validates that referenced dimensions/values exist
- System applies filter before tree construction
- System reports filtered data size

---

### Epic 2: Tree Construction

**E2: As a data analyst, I want to build a metric decomposition tree so that I can understand how different segments contribute to my metric.**

#### User Stories

**US-2.1: Greedy Feature Selection**
> As a data analyst, I want the system to select the best feature at each node so that my tree optimizes for my objective.

**Acceptance Criteria:**
- System computes metric for each candidate feature
- System selects based on max/min/closest-to-middle
- System handles ties via random selection
- System excludes features already in path

**US-2.2: Build Binary Tree**
> As a data analyst, I want a binary tree with TRUE/FALSE branches so that I maintain MECE properties.

**Acceptance Criteria:**
- Each node has exactly two children (or is a leaf)
- Left branch represents feature=FALSE
- Right branch represents feature=TRUE
- Sum of child metrics equals parent metric (within tolerance)

**US-2.3: Respect Max Depth**
> As a data analyst, I want to limit tree depth so that my tree remains interpretable.

**Acceptance Criteria:**
- Tree does not exceed configured max_depth
- Nodes at max_depth become leaves
- System reports actual depth achieved

**US-2.4: Handle Termination Conditions**
> As a data analyst, I want the tree to stop appropriately so that I don't get meaningless nodes.

**Acceptance Criteria:**
- Tree stops when max_depth reached
- Tree stops when no valid splits remain
- Tree creates leaf nodes at termination points

**US-2.5: Support Absence Mode**
> As a data analyst, I want to analyze by feature absence so that I can understand what's NOT contributing to my metric.

**Acceptance Criteria:**
- When absence=True, selection based on FALSE metric
- When absence=False, selection based on TRUE metric
- System documents equivalences in output

---

### Epic 3: Derived Metrics

**E3: As a data analyst, I want to use derived metrics so that I can analyze ratios and computed values.**

#### User Stories

**US-3.1: Parse Metric Formula**
> As a data analyst, I want to specify metrics as formulas so that I can analyze ratios like revenue/count.

**Acceptance Criteria:**
- System parses formulas like "revenue / count"
- System validates all referenced metrics exist
- System supports basic arithmetic (+, -, *, /)

**US-3.2: Compute Derived Metrics**
> As a data analyst, I want derived metrics computed correctly so that my ratios are accurate.

**Acceptance Criteria:**
- System computes SUM(numerator) / SUM(denominator) for each segment
- System handles division by zero gracefully
- System reports derived metric in outputs

---

### Epic 4: Frequent Itemset Integration

**E4: As a data analyst, I want to use frequent itemsets so that I can build non-greedy trees that capture multi-feature patterns.**

#### User Stories

**US-4.1: Configure Frequent Itemset Mining**
> As a data analyst, I want to configure itemset mining so that I control the optimization level.

**Acceptance Criteria:**
- opt_level > 1 enables itemset mining
- effective_opt_level decreases with depth
- System validates opt_level constraints

**US-4.2: Mine Itemsets with Retry**
> As a data analyst, I want the system to find itemsets even with sparse data so that the algorithm doesn't fail.

**Acceptance Criteria:**
- System starts with min_support=80%
- System halves min_support on empty results
- System stops at floor (0.01%) or max_retries
- System falls back to greedy if no itemsets found

**US-4.3: Build Spine from Itemset**
> As a data analyst, I want itemsets to create a path of nodes so that correlated features are captured together.

**Acceptance Criteria:**
- k-itemset creates k nodes in TRUE-branch spine
- Features ordered by metric (descending for max, ascending for min)
- FALSE branches filled recursively

**US-4.4: Support Absence Mode Itemset Discovery (Theta Sketch Only)**
> As a data engineer, I want to discover absence-based feature combinations so that I can analyze patterns defined by feature absence at scale.

**Acceptance Criteria:**
- System restricts absence mode with opt_level > 1 to Theta Sketch format only
- System raises clear error for Raw FACT/OLAP with absence=True and opt_level > 1
- System uses union coverage + De Morgan's law approach (NOT traditional FIS)
- For max_min_mean="min": System finds k presence features with maximum union coverage (yields rarest absence combination)
- For max_min_mean="max": System finds k presence features with minimum union coverage (yields most common absence combination)
- System orders selected features by individual absence metric for spine construction
- Algorithm complexity is O(n × k) avoiding combinatorial explosion

---

### Epic 5: Output Generation

**E5: As a data analyst, I want multiple output formats so that I can use the results in different contexts.**

#### User Stories

**US-5.1: Generate In-Memory Tree**
> As a developer, I want an in-memory tree object so that I can traverse and analyze programmatically.

**Acceptance Criteria:**
- Tree is returned as nested MDTNode objects
- All node attributes are populated
- Tree supports traversal methods

**US-5.2: Generate JSON Output**
> As a data analyst, I want JSON output so that I can save and share results.

**Acceptance Criteria:**
- Tree serializes to valid JSON
- Configuration included in output
- All node attributes included

**US-5.3: Generate Text Visualization**
> As a data analyst, I want a text tree diagram so that I can quickly understand the structure.

**Acceptance Criteria:**
- Output uses ASCII tree characters
- Each node shows metric, pct_of_root, pct_of_parent
- Output is readable in terminal/console

**US-5.4: Generate Graphviz Output**
> As a data analyst, I want graphviz output so that I can create publication-quality diagrams.

**Acceptance Criteria:**
- Output is valid DOT format
- Nodes labeled with feature and metrics
- Edges labeled with TRUE/FALSE

**US-5.5: Generate Plain English Narrative**
> As a data analyst, I want plain English descriptions so that I can feed results to an LLM for reasoning.

**Acceptance Criteria:**
- One paragraph per root-to-leaf path
- Includes absolute metric, pct_of_root, pct_of_parent
- Readable by non-technical stakeholders
- Suitable for LLM consumption

---

### Epic 6: Validation and Error Handling

**E6: As a data analyst, I want clear error messages so that I can fix configuration problems quickly.**

#### User Stories

**US-6.1: Validate Configuration**
> As a data analyst, I want configuration validated upfront so that I catch errors before processing.

**Acceptance Criteria:**
- All constraints checked before tree building
- Clear error messages for each violation
- Auto-corrections applied with warnings (e.g., md elevation)

**US-6.2: Validate Data**
> As a data analyst, I want data validated so that I know if my input is suitable.

**Acceptance Criteria:**
- Missing metrics raise clear errors
- Invalid pre_filter raises clear errors
- Dropped dimensions logged with reasons

**US-6.3: Handle Edge Cases**
> As a data analyst, I want edge cases handled gracefully so that the tool doesn't crash unexpectedly.

**Acceptance Criteria:**
- Same metric for all candidates → random selection
- No valid splits → leaf node
- Empty itemset results → retry then fallback

---

### Epic 7: Set Operations Abstraction Layer *(NEW in v1.1)*

**E7: As a data engineer, I want to choose between different set operation providers so that I can optimize for accuracy or scale based on my use case.**

#### User Stories

**US-7.1: Define SetOperationsProvider Interface**
> As a developer, I want a clean abstraction for set operations so that I can add new providers without modifying core logic.

**Acceptance Criteria:**
- Interface defines create, add, union, intersect, difference, get_estimate
- Interface includes is_exact() and get_error_bound() methods
- Interface includes serialize/deserialize for persistence

**US-7.2: Implement ThetaSketchProvider**
> As a data engineer, I want a Theta Sketch provider so that I can handle billions of records with approximate counts.

**Acceptance Criteria:**
- Provider wraps Apache DataSketches library
- is_exact() returns False
- get_error_bound() returns appropriate confidence interval
- Union operations complete in O(sketch_size) time

**US-7.3: Implement RoaringBitmapProvider**
> As a data analyst, I want a Roaring Bitmap provider so that I can get exact counts for audit and compliance.

**Acceptance Criteria:**
- Provider wraps pyroaring library
- is_exact() returns True
- get_error_bound() returns None
- Supports member query (contains method)

**US-7.4: Implement AUTO Provider Selection**
> As a data analyst, I want the system to automatically select the best provider so that I don't need to understand the tradeoffs.

**Acceptance Criteria:**
- AUTO mode selects based on use_case hint
- AUTO mode considers data cardinality
- AUTO mode warns if selection may be suboptimal
- User can override with explicit provider_type

**US-7.5: Add use_case Hints to MDTConfig**
> As a data analyst, I want to specify my use case so that the system can optimize for my needs.

**Acceptance Criteria:**
- Config accepts use_case parameter
- Valid use_cases: root_cause, data_quality, fraud, micro_segment, operational, missing_data
- use_case influences provider selection and output labeling

**US-7.6: Add Finding Classification**
> As a business user, I want findings labeled appropriately so that I understand what type of insight I'm seeing.

**Acceptance Criteria:**
- Findings classified as SEGMENT, DATA_ERROR, RISK, or OUTLIER
- Classification based on config and node metrics
- Plain English output uses appropriate terminology

**US-7.7: Update Output Generators with Finding Labels**
> As a data analyst, I want output formats to include finding types so that downstream systems can process them appropriately.

**Acceptance Criteria:**
- JSON output includes finding_type field
- Plain English narrative uses finding-specific templates
- Text tree shows finding type indicators

**US-7.8: Add DQ Violation Detection**
> As a data quality analyst, I want to find impossible combinations so that I can fix data errors.

**Acceptance Criteria:**
- System accepts constraint pairs (mutually exclusive features)
- System finds non-zero intersections using Roaring Bitmaps
- Output includes record IDs for investigation
- Severity classification (HIGH/MEDIUM/LOW)

---

## 15. Appendix

### A.1 Glossary

| Term | Definition |
|------|------------|
| **MECE** | Mutually Exclusive, Collectively Exhaustive - a property ensuring segments don't overlap and sum to total |
| **Theta Sketch** | A probabilistic data structure from Apache DataSketches for approximate distinct counting and set operations |
| **Roaring Bitmap** | A compressed bitmap data structure for exact set operations; ideal for audit and compliance use cases |
| **SetOperationsProvider** | Abstract interface allowing MDT to use different underlying data structures (Theta Sketch, Roaring Bitmap, etc.) |
| **Frequent Itemset** | A set of items (features) that appear together frequently in data |
| **opt_level** | Optimization level - determines the size of itemsets mined at each depth |
| **Presence** | Feature value is TRUE |
| **Absence** | Feature value is FALSE |
| **Spine** | The chain of nodes created along TRUE branches when using frequent itemsets |
| **Outlier** | A data point that differs significantly from other observations; valid but unusual |
| **Data Quality Issue** | Data that violates a hard constraint of reality or business logic; impossible and invalid |
| **Fraud** | Data that is technically valid but contextually suspicious; indicates deceptive intent |
| **Anomaly** | Umbrella term for any pattern that does not conform to expectations; requires investigation |

### A.2 Formula Syntax

```
moi ::= metric_name
      | metric_name operator metric_name
      | "(" moi ")"

operator ::= "+" | "-" | "*" | "/"

metric_name ::= [a-zA-Z_][a-zA-Z0-9_]*
```

Examples:
- `"revenue"`
- `"revenue / count"`
- `"clicks / impressions"`
- `"(revenue - cost) / revenue"`

### A.3 Pre-filter Syntax

```
pre_filter ::= condition
             | condition "&&" pre_filter

condition ::= dimension "=" value
            | dimension "!=" value

dimension ::= [a-zA-Z_][a-zA-Z0-9_]*
value ::= [a-zA-Z0-9_-]+
```

Examples:
- `"city=Mumbai"`
- `"city=Mumbai && gender=Male"`
- `"city=Mumbai && gender=Male && agegroup!=55-60"`

### A.4 Why Union Coverage for Absence Mode (Not Traditional FIS)

**The Problem with Traditional FIS for Negated Features:**

Frequent Itemset Mining algorithms (Apriori, FP-Growth) rely on the **anti-monotonicity property**:
- If itemset {A, B} is infrequent, then {A, B, C} is also infrequent
- This allows pruning of the search space

For **presence features**, this works:
- `city=Mumbai` has 5% support
- `city=Mumbai AND product=Luxury` has ≤ 5% support
- Support can only decrease as we add features

For **negated features**, support is inverted:
- `city=Mumbai` → 5% support
- `city≠Mumbai` → 95% support

This breaks FIS algorithms:
- Almost every negated feature has high support
- Almost every combination of negated features has high support
- No pruning possible → C(n,k) combinatorial explosion

**The Solution: Mathematical Equivalence via De Morgan's Law**

```
count(A≠v1 AND B≠v2 AND C≠v3) = Total - count(A=v1 OR B=v2 OR C=v3)
```

Finding a **rare absence combination** is equivalent to finding **presence features whose union covers most of the data**.

**Algorithm Comparison:**

| Aspect | Traditional FIS (Negated) | Union Coverage |
|--------|---------------------------|----------------|
| Complexity | O(2^n) — explosion | O(n × k) — greedy |
| Support threshold | Required, problematic | Not needed |
| Pruning | Not effective | N/A (greedy selection) |
| Theta Sketch ops | Intersection (problematic) | Union (natural) |

**Why Theta Sketches Excel at This:**

Theta Sketches support efficient union operations:
```python
# O(sketch_size) regardless of data size
union_sketch = sketch_A.union(sketch_B).union(sketch_C)
coverage = union_sketch.get_estimate()
absence_count = total - coverage
```

This makes the union coverage approach both mathematically sound and computationally efficient.

### A.5 Example Walkthrough

**Input:**
- Data: Sales transactions with dimensions [city, gender, agegroup, product_category] and metrics [revenue, count]
- Config: `md=3, moi="revenue", opt_level=2, max_min_mean="max"`

**Process:**

1. **Depth 0 (opt_level=2)**: Mine 2-itemsets
   - Best: {city=Mumbai, product_category=Electronics} with revenue $500K, $400K
   - Order: city=Mumbai ($500K) → product_category=Electronics ($400K)
   - Create spine: [city=Mumbai] —T→ [product_category=Electronics] —T→ leaf

2. **Depth 1 (city=Mumbai=F, opt_level=1)**: Greedy
   - Best: gender=Male with revenue $300K
   - Create: [gender=Male] with children

3. **Depth 2 (city=Mumbai=T, product_category=Electronics=F, opt_level=1)**: Greedy
   - Filter: city=Mumbai=T AND product_category=Electronics=F
   - Best: agegroup=25-34 with revenue $150K

4. Continue recursively...

**Output:** Complete tree with all paths documented in plain English.

---

## Document Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Owner | Vijay | | |
| Technical Lead | | | |
| QA Lead | | | |

---

*End of Document*
