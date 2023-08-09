#! /bin/bash
subject=sub-02

export SURF_DIR=/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives/freesurfer/$subject
echo $SURF_DIR

python -m neuropythy atlas $subject --verbose 

for hemi in lh rh
do
	# mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_eccen.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/${hemi}.benson14_eccen 
	# mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_sigma.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/${hemi}.benson14_sigma 
	# mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_angle.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/${hemi}.benson14_angle 
	mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_varea.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/${hemi}.benson14_varea 
	mri_surf2surf --srcsubject fsaverage --trgsubject $subject --hemi ${hemi} \
    --sval ${SURF_DIR}/surf/${hemi}.benson14_varea.mgz --tval ${SURF_DIR}/surf/${hemi}.benson14_varea_native.mgz
done


# # see: https://github.com/noahbenson/neuropythy
# # https://nben.net/Retinotopy-Tutorial/#atlas-generation
# # conda activate dag_atlas
# # python -m neuropythy benson14_retinotopy <sub> -v
# # From Mayra
