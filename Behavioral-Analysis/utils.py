import re
import numpy as np

def associate_items_to_poses():
    with open('/Users/maxalex/Desktop/Questionnaire_3D_body_postures (2).txt') as f:
        txt = f.read()

    poses = {}

    re_stim = "Inizio blocco: Stim_uparam_[0-9]*_Viewpoint_[1-3]_scale_[0-9]{1,2}"
    while bool(re.search(re_stim, txt)):
        stim_eval = re.search(re_stim, txt)
        pose_name = stim_eval.group()[15:]
        txt = txt[stim_eval.end():].strip()

        # Get text block until next stimuli
        block = txt[:txt.find(f'Fine blocco: Stim_uparam_')]
        items = []
        regex = "Q([0-9]*[.])?[0-9]+"  # In the second half Q's are coded diff
        while f'QID' in block or bool(re.search(regex, block)):
            if f'QID' in block:
                block = block[block.find(f'QID'):].strip()
            else:
                block = block[re.search(regex, block).start():].strip()

            items.append(block[:block.find(f' ')])
            block = block[block.find(f' '):]

        poses[pose_name] = items

    np.save(f'items-questions.npy', poses)


if __name__ == "__main__":
    associate_items_to_poses()
