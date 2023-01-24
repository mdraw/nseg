```.
├── fib25
│   ├── testing
│   │   ├── ground_truth
│   │   │   └── data.n5
│   │   │       └── volumes
│   │   │           ├── mask
│   │   │           ├── neuron_ids
│   │   │           ├── raw
│   │   │           ├── roi_1_relabelled_ids
│   │   │           └── roi_2_relabelled_ids
│   │   └── segmentations
│   │       └── data.zarr
│   │           ├── rags
│   │           │   ├── ACLSD
│   │           │   │   └── fib25_auto_basic_300k_testing
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   ├── ACRLSD
│   │           │   │   └── fib25_auto_full_300k_testing
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   ├── Baseline
│   │           │   │   └── fib25_vanilla_400k_testing
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   ├── LongRange
│   │           │   │   └── fib25_longrange_400k_testing
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   ├── MALIS
│   │           │   │   └── fib25_malis_200k_testing
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   └── MTLSD
│   │           │       └── fib25_mtlsd_400k_testing
│   │           │           ├── edges_hist_quant_75.bson
│   │           │           ├── edges_hist_quant_75.metadata.json
│   │           │           ├── nodes.bson
│   │           │           └── nodes.metadata.json
│   │           └── volumes
│   │               ├── ACLSD
│   │               │   └── fragments
│   │               ├── ACRLSD
│   │               │   └── fragments
│   │               ├── Baseline
│   │               │   └── fragments
│   │               ├── FFN
│   │               │   ├── roi_1_relabelled_seg
│   │               │   ├── roi_2_relabelled_seg
│   │               │   └── segmentation
│   │               ├── LongRange
│   │               │   └── fragments
│   │               ├── MALIS
│   │               │   └── fragments
│   │               └── MTLSD
│   │                   └── fragments
│   └── training
│       ├── trvol-250-1.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       ├── trvol-250-2.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       ├── tstvol-520-1.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       └── tstvol-520-2.zarr
│           └── volumes
│               ├── labels
│               │   └── neuron_ids
│               └── raw
├── hemi
│   ├── testing
│   │   ├── ground_truth
│   │   │   └── data.zarr
│   │   │       └── volumes
│   │   │           ├── mask
│   │   │           ├── roi_1
│   │   │           │   ├── consolidated_ids
│   │   │           │   ├── neuron_ids
│   │   │           │   └── raw
│   │   │           ├── roi_2
│   │   │           │   ├── consolidated_ids
│   │   │           │   ├── neuron_ids
│   │   │           │   └── raw
│   │   │           └── roi_3
│   │   │               ├── consolidated_ids
│   │   │               ├── neuron_ids
│   │   │               └── raw
│   │   └── segmentations
│   │       └── data.zarr
│   │           ├── rags
│   │           │   ├── ACLSD
│   │           │   │   ├── hemi_affs_from_lsd_200k_roi_1
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   ├── hemi_affs_from_lsd_200k_roi_2
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   └── hemi_affs_from_lsd_200k_roi_3
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   ├── ACRLSD
│   │           │   │   ├── hemi_affs_from_lsd_full_200k_roi_1
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   ├── hemi_affs_from_lsd_full_200k_roi_2
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   └── hemi_affs_from_lsd_full_200k_roi_3
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   ├── Baseline
│   │           │   │   ├── hemi_vanilla_400k_roi_1
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   ├── hemi_vanilla_400k_roi_2
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   └── hemi_vanilla_400k_roi_3
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   ├── LongRange
│   │           │   │   ├── hemi_longrange_400k_roi_1
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   ├── hemi_longrange_400k_roi_2
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   └── hemi_longrange_400k_roi_3
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   ├── MALIS
│   │           │   │   ├── hemi_malis_400k_roi_1
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   ├── hemi_malis_400k_roi_2
│   │           │   │   │   ├── edges_hist_quant_75.bson
│   │           │   │   │   ├── edges_hist_quant_75.metadata.json
│   │           │   │   │   ├── nodes.bson
│   │           │   │   │   └── nodes.metadata.json
│   │           │   │   └── hemi_malis_400k_roi_3
│   │           │   │       ├── edges_hist_quant_75.bson
│   │           │   │       ├── edges_hist_quant_75.metadata.json
│   │           │   │       ├── nodes.bson
│   │           │   │       └── nodes.metadata.json
│   │           │   └── MTLSD
│   │           │       ├── hemi_mtlsd_400k_roi_1_no_eps
│   │           │       │   ├── edges_hist_quant_75.bson
│   │           │       │   ├── edges_hist_quant_75.metadata.json
│   │           │       │   ├── nodes.bson
│   │           │       │   └── nodes.metadata.json
│   │           │       ├── hemi_mtlsd_400k_roi_2_no_eps
│   │           │       │   ├── edges_hist_quant_75.bson
│   │           │       │   ├── edges_hist_quant_75.metadata.json
│   │           │       │   ├── nodes.bson
│   │           │       │   └── nodes.metadata.json
│   │           │       └── hemi_mtlsd_400k_roi_3_no_eps
│   │           │           ├── edges_hist_quant_75.bson
│   │           │           ├── edges_hist_quant_75.metadata.json
│   │           │           ├── nodes.bson
│   │           │           └── nodes.metadata.json
│   │           └── volumes
│   │               ├── ACLSD
│   │               │   ├── roi_1
│   │               │   │   └── fragments
│   │               │   ├── roi_2
│   │               │   │   └── fragments
│   │               │   └── roi_3
│   │               │       └── fragments
│   │               ├── ACRLSD
│   │               │   ├── roi_1
│   │               │   │   └── fragments
│   │               │   ├── roi_2
│   │               │   │   └── fragments
│   │               │   └── roi_3
│   │               │       └── fragments
│   │               ├── Baseline
│   │               │   ├── roi_1
│   │               │   │   └── fragments
│   │               │   ├── roi_2
│   │               │   │   └── fragments
│   │               │   └── roi_3
│   │               │       └── fragments
│   │               ├── FFN
│   │               │   ├── roi_1
│   │               │   │   ├── consolidated_ids
│   │               │   │   └── neuron_ids
│   │               │   ├── roi_2
│   │               │   │   ├── consolidated_ids
│   │               │   │   └── neuron_ids
│   │               │   └── roi_3
│   │               │       ├── consolidated_ids
│   │               │       └── neuron_ids
│   │               ├── LongRange
│   │               │   ├── roi_1
│   │               │   │   └── fragments
│   │               │   ├── roi_2
│   │               │   │   └── fragments
│   │               │   └── roi_3
│   │               │       └── fragments
│   │               ├── MALIS
│   │               │   ├── roi_1
│   │               │   │   └── fragments
│   │               │   ├── roi_2
│   │               │   │   └── fragments
│   │               │   └── roi_3
│   │               │       └── fragments
│   │               └── MTLSD
│   │                   ├── roi_1
│   │                   │   └── fragments
│   │                   ├── roi_2
│   │                   │   └── fragments
│   │                   └── roi_3
│   │                       └── fragments
│   └── training
│       ├── eb-inner-groundtruth-with-context-x20172-y2322-z14332.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       ├── eb-outer-groundtruth-with-context-x20532-y3512-z14332.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       ├── fb-inner-groundtruth-with-context-x17342-y4052-z14332.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       ├── fb-outer-groundtruth-with-context-x13542-y2462-z14332.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       ├── lh-groundtruth-with-context-x7737-y20781-z12444.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       ├── lobula-groundtruth-with-context-x3648-y12800-z29056.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       ├── pb-groundtruth-with-context-x8472-y2372-z9372.zarr
│       │   └── volumes
│       │       ├── labels
│       │       │   └── neuron_ids
│       │       └── raw
│       └── pb-groundtruth-with-context-x8472-y2892-z9372.zarr
│           └── volumes
│               ├── labels
│               │   └── neuron_ids
│               └── raw
└── zebrafinch
    ├── testing
    │   ├── ground_truth
    │   │   ├── data.zarr
    │   │   │   └── volumes
    │   │   │       └── neuropil_mask
    │   │   ├── testing
    │   │   │   ├── consolidated
    │   │   │   │   └── zebrafinch_gt_skeletons_new_gt_9_9_20_testing
    │   │   │   │       ├── nodes_zebrafinch_components_11_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_11_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_18_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_18_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_25_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_25_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_32_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_32_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_40_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_40_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_47_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_47_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_54_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_54_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_61_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_61_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_68_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_68_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_76_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_76_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_components_benchmark_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_components_benchmark_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_11_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_11_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_18_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_18_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_25_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_25_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_32_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_32_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_40_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_40_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_47_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_47_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_54_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_54_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_61_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_61_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_68_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_68_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_76_micron_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_76_micron_roi_masked.metadata.json
    │   │   │   │       ├── nodes_zebrafinch_mask_benchmark_roi_masked.bson
    │   │   │   │       ├── nodes_zebrafinch_mask_benchmark_roi_masked.metadata.json
    │   │   │   │       ├── zebrafinch.edges.bson
    │   │   │   │       ├── zebrafinch.edges.metadata.json
    │   │   │   │       ├── zebrafinch.nodes.bson
    │   │   │   │       └── zebrafinch.nodes.metadata.json
    │   │   │   └── original
    │   │   │       ├── jsons
    │   │   │       │   ├── 116.json
    │   │   │       │   ├── 133.json
    │   │   │       │   ├── 164.json
    │   │   │       │   ├── 170.json
    │   │   │       │   ├── 182.json
    │   │   │       │   ├── 187.json
    │   │   │       │   ├── 190.json
    │   │   │       │   ├── 214.json
    │   │   │       │   ├── 227.json
    │   │   │       │   ├── 233.json
    │   │   │       │   ├── 235.json
    │   │   │       │   ├── 246.json
    │   │   │       │   ├── 261.json
    │   │   │       │   ├── 266.json
    │   │   │       │   ├── 310.json
    │   │   │       │   ├── 321.json
    │   │   │       │   ├── 354.json
    │   │   │       │   ├── 358.json
    │   │   │       │   ├── 364.json
    │   │   │       │   ├── 371.json
    │   │   │       │   ├── 397.json
    │   │   │       │   ├── 402.json
    │   │   │       │   ├── 404.json
    │   │   │       │   ├── 416.json
    │   │   │       │   ├── 423.json
    │   │   │       │   ├── 429.json
    │   │   │       │   ├── 441.json
    │   │   │       │   ├── 469.json
    │   │   │       │   ├── 480.json
    │   │   │       │   ├── 496.json
    │   │   │       │   ├── 508.json
    │   │   │       │   ├── 530.json
    │   │   │       │   ├── 545.json
    │   │   │       │   ├── 547.json
    │   │   │       │   ├── 557.json
    │   │   │       │   ├── 582.json
    │   │   │       │   ├── 588.json
    │   │   │       │   ├── 607.json
    │   │   │       │   ├── 670.json
    │   │   │       │   └── 675.json
    │   │   │       └── nmls
    │   │   │           ├── test_set_skeleton_116.nml
    │   │   │           ├── test_set_skeleton_133.nml
    │   │   │           ├── test_set_skeleton_164.nml
    │   │   │           ├── test_set_skeleton_170.nml
    │   │   │           ├── test_set_skeleton_182.nml
    │   │   │           ├── test_set_skeleton_187.nml
    │   │   │           ├── test_set_skeleton_190.nml
    │   │   │           ├── test_set_skeleton_1.nml
    │   │   │           ├── test_set_skeleton_214.nml
    │   │   │           ├── test_set_skeleton_227.nml
    │   │   │           ├── test_set_skeleton_233.nml
    │   │   │           ├── test_set_skeleton_235.nml
    │   │   │           ├── test_set_skeleton_246.nml
    │   │   │           ├── test_set_skeleton_261.nml
    │   │   │           ├── test_set_skeleton_266.nml
    │   │   │           ├── test_set_skeleton_29.nml
    │   │   │           ├── test_set_skeleton_2.nml
    │   │   │           ├── test_set_skeleton_310.nml
    │   │   │           ├── test_set_skeleton_321.nml
    │   │   │           ├── test_set_skeleton_354.nml
    │   │   │           ├── test_set_skeleton_358.nml
    │   │   │           ├── test_set_skeleton_364.nml
    │   │   │           ├── test_set_skeleton_371.nml
    │   │   │           ├── test_set_skeleton_397.nml
    │   │   │           ├── test_set_skeleton_402.nml
    │   │   │           ├── test_set_skeleton_404.nml
    │   │   │           ├── test_set_skeleton_40.nml
    │   │   │           ├── test_set_skeleton_416.nml
    │   │   │           ├── test_set_skeleton_423.nml
    │   │   │           ├── test_set_skeleton_429.nml
    │   │   │           ├── test_set_skeleton_43.nml
    │   │   │           ├── test_set_skeleton_441.nml
    │   │   │           ├── test_set_skeleton_469.nml
    │   │   │           ├── test_set_skeleton_480.nml
    │   │   │           ├── test_set_skeleton_496.nml
    │   │   │           ├── test_set_skeleton_508.nml
    │   │   │           ├── test_set_skeleton_530.nml
    │   │   │           ├── test_set_skeleton_545.nml
    │   │   │           ├── test_set_skeleton_547.nml
    │   │   │           ├── test_set_skeleton_557.nml
    │   │   │           ├── test_set_skeleton_582.nml
    │   │   │           ├── test_set_skeleton_588.nml
    │   │   │           ├── test_set_skeleton_5.nml
    │   │   │           ├── test_set_skeleton_607.nml
    │   │   │           ├── test_set_skeleton_62.nml
    │   │   │           ├── test_set_skeleton_66.nml
    │   │   │           ├── test_set_skeleton_670.nml
    │   │   │           ├── test_set_skeleton_675.nml
    │   │   │           ├── test_set_skeleton_73.nml
    │   │   │           └── test_set_skeleton_99.nml
    │   │   └── validation
    │   │       ├── consolidated
    │   │       │   └── zebrafinch_gt_skeletons_new_gt_9_9_20_validation
    │   │       │       ├── nodes_zebrafinch_components_11_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_11_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_components_18_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_18_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_components_25_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_25_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_components_32_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_32_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_components_40_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_40_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_components_47_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_47_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_components_54_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_54_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_components_61_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_61_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_components_76_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_76_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_components_benchmark_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_components_benchmark_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_11_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_11_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_18_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_18_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_25_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_25_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_32_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_32_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_40_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_40_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_47_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_47_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_54_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_54_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_61_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_61_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_76_micron_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_76_micron_roi_masked.metadata.json
    │   │       │       ├── nodes_zebrafinch_mask_benchmark_roi_masked.bson
    │   │       │       ├── nodes_zebrafinch_mask_benchmark_roi_masked.metadata.json
    │   │       │       ├── zebrafinch.edges.bson
    │   │       │       ├── zebrafinch.edges.metadata.json
    │   │       │       ├── zebrafinch.nodes.bson
    │   │       │       └── zebrafinch.nodes.metadata.json
    │   │       └── original
    │   │           ├── jsons
    │   │           └── nmls
    │   │               ├── 0005.nml
    │   │               ├── 0019.nml
    │   │               ├── 0021.nml
    │   │               ├── 0088.nml
    │   │               ├── 0105.nml
    │   │               ├── 0115.nml
    │   │               ├── 0219.nml
    │   │               ├── 0253.nml
    │   │               ├── 0270.nml
    │   │               ├── 0390.nml
    │   │               ├── 1111.nml
    │   │               └── 2333.nml
    │   └── segmentations
    │       └── data.zarr
    │           ├── rags
    │           │   ├── ACLSD
    │           │   │   └── zebrafinch_auto_basic_163k_testing_masked_ffn
    │           │   │       ├── edges_hist_quant_75.bson
    │           │   │       ├── edges_hist_quant_75.metadata.json
    │           │   │       ├── nodes.bson
    │           │   │       └── nodes.metadata.json
    │           │   ├── ACRLSD
    │           │   │   └── zebrafinch_auto_full_190k_testing_masked_ffn
    │           │   │       ├── edges_hist_quant_75.bson
    │           │   │       ├── edges_hist_quant_75.metadata.json
    │           │   │       ├── nodes.bson
    │           │   │       └── nodes.metadata.json
    │           │   ├── Baseline
    │           │   │   └── zebrafinch_vanilla_400k_testing_masked_ffn
    │           │   │       ├── edges_hist_quant_75.bson
    │           │   │       ├── edges_hist_quant_75.metadata.json
    │           │   │       ├── nodes.bson
    │           │   │       └── nodes.metadata.json
    │           │   ├── LongRange
    │           │   │   └── zebrafinch_longrange_padding_400k_testing_masked_ffn
    │           │   │       ├── edges_hist_quant_75.bson
    │           │   │       ├── edges_hist_quant_75.metadata.json
    │           │   │       ├── nodes.bson
    │           │   │       └── nodes.metadata.json
    │           │   ├── MALIS
    │           │   │   └── zebrafinch_malis_400k_testing_masked_ffn
    │           │   │       ├── edges_hist_quant_75.bson
    │           │   │       ├── edges_hist_quant_75.metadata.json
    │           │   │       ├── nodes.bson
    │           │   │       └── nodes.metadata.json
    │           │   └── MTLSD
    │           │       └── zebrafinch_mtlsd_400k_testing_masked_ffn
    │           │           ├── edges_hist_quant_75.bson
    │           │           ├── edges_hist_quant_75.metadata.json
    │           │           ├── nodes.bson
    │           │           └── nodes.metadata.json
    │           └── volumes
    │               ├── ACLSD
    │               │   └── fragments
    │               ├── ACRLSD
    │               │   └── fragments
    │               ├── Baseline
    │               │   └── fragments
    │               ├── FFN
    │               │   ├── 11_micron_roi
    │               │   ├── 18_micron_roi
    │               │   ├── 25_micron_roi
    │               │   ├── 32_micron_roi
    │               │   ├── 40_micron_roi
    │               │   ├── 47_micron_roi
    │               │   ├── 54_micron_roi
    │               │   ├── 61_micron_roi
    │               │   ├── 68_micron_roi
    │               │   ├── 76_micron_roi
    │               │   └── benchmark_roi
    │               ├── LongRange
    │               │   └── fragments
    │               ├── MALIS
    │               │   └── fragments
    │               └── MTLSD
    │                   └── fragments
    └── training
        ├── gt_z255-383_y1407-1663_x1535-1791.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z2559-2687_y4991-5247_x4863-5119.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z2815-2943_y5631-5887_x4607-4863.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z2834-2984_y5311-5461_x5077-5227.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z2868-3018_y5744-5894_x5157-5307.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z2874-3024_y5707-5857_x5304-5454.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z2934-3084_y5115-5265_x5140-5290.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3096-3246_y5954-6104_x5813-5963.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3118-3268_y6538-6688_x6100-6250.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3126-3276_y6857-7007_x5694-5844.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3436-3586_y599-749_x2779-2929.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3438-3588_y2775-2925_x3476-3626.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3456-3606_y3188-3338_x4043-4193.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3492-3642_y7888-8038_x8374-8524.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3492-3642_y841-991_x381-531.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3596-3746_y3888-4038_x3661-3811.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3604-3754_y4101-4251_x3493-3643.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3608-3758_y3829-3979_x3423-3573.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3702-3852_y9605-9755_x2244-2394.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3710-3860_y8691-8841_x2889-3039.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3722-3872_y4548-4698_x2879-3029.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3734-3884_y4315-4465_x2209-2359.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z3914-4064_y9035-9185_x2573-2723.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z4102-4252_y6330-6480_x1899-2049.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z4312-4462_y9341-9491_x2419-2569.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z4440-4590_y7294-7444_x2350-2500.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z4801-4951_y10154-10304_x1972-2122.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z4905-5055_y928-1078_x1729-1879.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z4951-5101_y9415-9565_x2272-2422.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z5001-5151_y9426-9576_x2197-2347.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z5119-5247_y1023-1279_x1663-1919.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        ├── gt_z5405-5555_y10490-10640_x3406-3556.zarr
        │   └── volumes
        │       ├── labels
        │       │   └── neuron_ids
        │       └── raw
        └── gt_z734-884_y9561-9711_x563-713.zarr
            └── volumes
                ├── labels
                │   └── neuron_ids
                └── raw

426 directories, 314 files```
