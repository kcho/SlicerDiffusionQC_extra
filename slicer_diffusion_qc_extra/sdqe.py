#!/env/python

# coding: utf-8

# # Diffusion QC summary
# 
# ## Summarizing `Slicer DiffusionQC` outputs
# 
# Friday, March 20, 2020
# 
# Kcho

from pathlib import Path
import pandas as pd
import numpy as np
import re
import nibabel as nb
import matplotlib.pyplot as plt


class QcDataDir:
    """Individual QC output directory object"""
    def __init__(self, qc_dir):
        """Set the file paths from the directories"""
        self.qc_dir = Path(qc_dir)
        
        # get the overlapping strings in all file names inside the qc_dir
        self.name_prefix = list(self.qc_dir.glob('*.raw'))[0].name[:-4]
    
        # setting attributes for all files
        self.all_files = self.qc_dir.glob('*')
        for file in self.all_files:
            file_name = file.name
            unique_string = file_name.split(self.name_prefix)[1]
            
            var_name = '_'.join(unique_string.split('.'))[1:]
            setattr(self, var_name, file)
            
            if var_name.endswith('npy'):
                setattr(self, re.sub('npy', 'array', var_name), np.load(file))
                
            if var_name.endswith('csv'):
                setattr(self, re.sub('csv', 'df', var_name), pd.read_csv(file))
        
        self._rearrange_dfs()
        
    def _rearrange_dfs(self):
        self.QC_df.columns = ['Gradient', 'QC', 'b value']
        self.confidence_df.columns = ['Gradient', 'Sure', 'b value']
        
    def plot_KL_divergence(self):
        fig, ax = plt.subplots(ncols=1, figsize=(10, 5), dpi=150)
        img = ax.imshow(self.KLdiv_array)
        ax.set_ylabel('Volume')
        ax.set_xlabel('Z Slice')
        ax.set_title(f'KL divergence: {self.qc_dir.name}')
        fig.colorbar(img)
        fig.show()
        
        
class QcStudyDir:
    def __init__(self, qc_study_dir):
        self.qc_study_dir = Path(qc_study_dir)
        
        self.subdir_paths = [x for x in self.qc_study_dir.glob('*') if x.is_dir()]
        self.subdir_names = [x.name for x in self.subdir_paths]
        
        self.qcDataDirs = []
        self.df = pd.DataFrame(columns=['registered'])
        for subdir_path in self.subdir_paths:
            try:
                qcDataDir = QcDataDir(subdir_path)
                self.qcDataDirs.append(qcDataDir)
                self.df.loc[subdir_path.name, 'registered'] = 1
            except:
                self.qcDataDirs.append(0)
                self.df.loc[subdir_path.name, 'registered'] = 0
            
        self.collect_arrays()
        
    def collect_arrays(self):
        self.KLdiv_array_all = np.stack([x.KLdiv_array for x in self.qcDataDirs if x!=0], axis=2)
        self.QC_array_all = np.stack([x.QC_array for x in self.qcDataDirs if x!=0], axis=1)
        self.confidence_array_all = np.stack([x.confidence_array for x in self.qcDataDirs if x!=0], axis=1)
        
        
    def get_failure_info(self, dwi_study_dir):
        failed_subjects = self.df[self.df.registered == 0].index

        for failed_subject in failed_subjects:
            dwi_loc = Path(dwi_study_dir) / failed_subject / 'dwi' / (failed_subject + '_dwi.nii.gz')
            if dwi_loc.is_file():
                dwi_img = nb.load(str(dwi_loc))
                self.df.loc[failed_subject, 'shape'] = str(dwi_img.shape)
            else:
                self.df.loc[failed_subject, 'files'] = ' '.join(list(dwi_loc.parent.glob('*')))

    def plot_KL_divergence_for_all(self):
        fig, ax = plt.subplots(ncols=1, figsize=(10, 5), dpi=150)
        KLdiv_mean = np.mean(self.KLdiv_array_all, axis=2)
        img = ax.imshow(KLdiv_mean)
        ax.set_ylabel('Volume')
        ax.set_xlabel('Z Slice')
        ax.set_title(f'Average KL divergence\n{self.KLdiv_array_all.shape[-1]} subjects')
        fig.colorbar(img)
        fig.show()
 
    def plot_KL_divergence_for_a_subject(self, subject_id):
        fig, ax = plt.subplots(ncols=1, figsize=(10, 5), dpi=150)
        subject_index = self.subdir_names.index(subject_id)
        subject_KLdiv = self.qcDataDirs[subject_index].KLdiv_array
        img = ax.imshow(subject_KLdiv)
        ax.set_ylabel('Volume')
        ax.set_xlabel('Z Slice')
        ax.set_title(f'KL divergence: {subject_id}')
        fig.colorbar(img)
        fig.show()

        
    def plot_qc_confidence_for_all(self):
        fig, ax = plt.subplots(ncols=1, figsize=(10, 5), dpi=150)
        qc_img = ax.imshow(self.QC_array_all, cmap='gray', alpha=0.5)
        conf_img = ax.imshow(self.confidence_array_all, cmap='autumn', alpha=0.4)
        ax.set_ylabel('Volume')
        ax.set_xlabel('Subjects')
        ax.set_title('DiffusionQC Summary')

        fig.subplots_adjust(right=0.9)

        d = ax.get_position()
        cax = fig.add_axes([0.92, d.get_points()[0][1], 0.02, d.height])
        fig.colorbar(conf_img, cax=cax, ticks=[1,0])
        cax.set_yticklabels(['Confident', 'not confident'])

        cax = fig.add_axes([0.8, d.get_points()[1][1] + 0.02, 0.1, 0.02])
        fig.colorbar(qc_img, cax=cax, ticks=[1,0], orientation='horizontal')
        cax.tick_params(bottom=False, top=True)
        cax.tick_params(labelbottom=False, labeltop=True)
        cax.set_xticklabels(['Pass', 'Fail'])

        fig.show()
        
    def plot_failed_volume_count_for_all(self):
        failed_volumes = np.sum(self.QC_array_all==0, axis=0)
        low_confident_volumes = np.sum(self.confidence_array_all==0, axis=0)

        fig, axes = plt.subplots(nrows=2, figsize=(10, 5), dpi=150)
        axes[0].plot(failed_volumes, 'ko')
        axes[0].set_title('Number of failed volumes across subjects')
        axes[1].plot(low_confident_volumes, 'ko')
        axes[1].set_title('Number of low confident volumes across subjects')

        for ax in np.ravel(axes):
            ax.set_ylabel('Number of volumes')
            ax.set_xlabel('Subjects')

        fig.subplots_adjust(hspace=0.5)
        fig.show()


def examples():
    qcStudyDir = QcStudyDir('/QC/DIR/FOR/STUDY')
    rawDataDir = '/RAW/DIR/FOR/STUDY'

    # specific to tokyo data
    qcStudyDir.get_failure_info(rawDataDir)

    qcStudyDir.plot_KL_divergence_for_all()
    qcStudyDir.plot_qc_confidence_for_all()
    qcStudyDir.plot_failed_volume_count_for_all()

    # print df
    print(qcStudyDir.df)

    # plot 
