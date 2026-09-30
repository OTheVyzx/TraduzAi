"""Connected layout stays geometric even with legacy Ollama settings."""
from unittest.mock import patch
import numpy as np
from layout.balloon_layout import _derive_connected_visual_boxes


def test_connected_layout_never_invokes_ollama_with_legacy_enabled_settings():
    image=np.full((120,220,3),255,dtype=np.uint8)
    lobes=[[10,10,100,100],[110,10,210,110]]
    groups=[[20,20,80,80],[120,20,190,90]]
    positions=[[15,15,95,95],[115,15,205,105]]
    with patch('layout.balloon_layout._score_subregion_quality',return_value=.6), \
         patch('layout.balloon_layout._derive_connected_text_groups',return_value=(groups,.5)), \
         patch('layout.balloon_layout._derive_connected_position_bboxes',return_value=positions), \
         patch('layout.balloon_layout._refine_connected_position_bboxes_with_ollama',return_value=None) as remote:
        result=_derive_connected_visual_boxes(image,[10,10,210,110],[0,0,220,120],lobes,
            'left-right',{'enabled':True,'provider':'ollama'})
    remote.assert_not_called()
    assert result['connected_position_bboxes']==positions
    assert result['connected_lobe_bboxes']==lobes
    assert result['connected_text_groups']==groups
    assert result['connected_position_reasoner']=='heuristic'
    assert result['connected_reasoner_model']==''
