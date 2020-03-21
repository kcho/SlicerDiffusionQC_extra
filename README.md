# Slicer Diffusion QC extra

## Summarizes outputs from `SlicerDiffusionQC`


### Import library
```sh
git clone https://github.com/kcho/SlicerDiffusionQC_extra
```



### Import library
```py
from SlicerDiffusionQC_extra.slicer_diffusion_qc_extra.sdqe import QcDataDir, QcStudyDir
```



### Loading QC outputs from a single subject


```py
data_qc_dir = '/QC/DIR/FOR/SUBJECT_1'

subjQc = QcDataDir(data_qc_dir)

print(subjQc.QC_df)
print(subjQc.confidence_df)

subjQc.plot_KL_divergence()
```



### Loading QC outputs from more than one subject


```py
qcStudyDir = QcStudyDir('/QC/DIR/FOR/STUDY')
rawDataDir = '/RAW/DIR/FOR/STUDY'

# specific to datasets
qcStudyDir.get_failure_info(rawDataDir)

qcStudyDir.plot_KL_divergence_for_all()
qcStudyDir.plot_qc_confidence_for_all()
qcStudyDir.plot_failed_volume_count_for_all()

# print df
print(qcStudyDir.df)

# plot 
qcStudyDir.plot_KL_divergence_for_a_subject('sub-1')
qcStudyDir.plot_KL_divergence_for_a_subject('sub-2')
```

