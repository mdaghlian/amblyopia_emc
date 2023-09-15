#! /bin/bash
export FREESURFER_HOME=/packages/freesurfer/7.3.2 
export FS_LICENSE=/data1/projects/dumoulinlab/Lab_members/Kathi/programs/freesurfer-license/license.txt
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives/freesurfer
subs=("sub-01" "sub-02")
for subject in "${subs[@]}"; do
    # Check
    export SURF_DIR=/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives/freesurfer/$subject
    # Use find to check if any file contains "benson14" in the SURF_DIR
    if find "${SURF_DIR}/surf" -type f -name "*benson14*mgz" -print -quit | grep -q .; then
        echo "Found 'benson14' file for $subject"
    else
        echo "No 'benson14' file found in $subject"
        python -m neuropythy atlas $subject --verbose 
    fi

    # Check if "b14" folder exists, if not, create it
    if [ ! -d "${SURF_DIR}/label/b14" ]; then
        mkdir "${SURF_DIR}/label/b14"
        echo "Created 'b14' label folder for $subject"
    fi
    # Now convert to labels
    for hemi in lh rh
    do
        mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_eccen.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/b14/${hemi}.benson14_eccen 
        mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_sigma.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/b14/${hemi}.benson14_sigma 
        mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_angle.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/b14/${hemi}.benson14_angle 
        mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_varea.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/b14/${hemi}.benson14_varea
        # mri_surf2surf --srcsubject fsaverage --trgsubject $subject --hemi ${hemi} --sval ${SURF_DIR}/surf/${hemi}.benson14_varea.mgz --tval ${SURF_DIR}/surf/${hemi}.benson14_varea_native.mgz
    done


done





# #! /bin/bash
# subject=sub-02

# export SURF_DIR=/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives/freesurfer/$subject
# echo $SURF_DIR

# python -m neuropythy atlas $subject --verbose 

# for hemi in lh rh
# do
# 	# mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_eccen.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/${hemi}.benson14_eccen 
# 	# mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_sigma.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/${hemi}.benson14_sigma 
# 	# mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_angle.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/${hemi}.benson14_angle 
# 	mri_surfcluster --in ${SURF_DIR}/surf/${hemi}.benson14_varea.mgz --subject $subject --hemi ${hemi} --thmin 0 --sign pos --no-adjust --olab ${SURF_DIR}/label/${hemi}.benson14_varea 
# 	mri_surf2surf --srcsubject fsaverage --trgsubject $subject --hemi ${hemi} \
#     --sval ${SURF_DIR}/surf/${hemi}.benson14_varea.mgz --tval ${SURF_DIR}/surf/${hemi}.benson14_varea_native.mgz
# done


# # # see: https://github.com/noahbenson/neuropythy
# # # https://nben.net/Retinotopy-Tutorial/#atlas-generation
# # # conda activate dag_atlas
# # # python -m neuropythy benson14_retinotopy <sub> -v
# # # From Mayra
