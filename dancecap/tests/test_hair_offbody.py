"""avatar_hair.offbody_hair + hair_maps(extra_hair, whip_mult): motion-blurred hair whip that the dancers-stage split
labelled body (larissa_kadu_38 review/dancers_body_hair.jpg) is excluded from the follow's body texels."""
import numpy as np

from dancecap import avatar_hair as ah


def _scene():
    H, W = 240, 240
    full = np.zeros((H, W), np.uint8)
    full[60:230, 90:150] = 1           # torso + legs
    full[20:60, 100:140] = 1           # head
    full[30:50, 140:220] = 1           # hair whip to the right of the head (outer part smeared)
    full[150:165, 40:90] = 1           # her arm, which the model put a little higher (misfit, not hair)
    sil = np.zeros((H, W), np.uint8)   # posed model silhouette: body, head and the arm (shifted)
    sil[60:230, 90:150] = 1
    sil[20:60, 100:140] = 1
    sil[140:152, 40:90] = 1
    hair = np.zeros((H, W), np.uint8)  # the split found only the inner part of the whip
    hair[30:50, 140:165] = 1
    return full, sil, hair


def test_offbody_hair_takes_the_smeared_whip_but_not_a_misfit_limb():
    full, sil, hair = _scene()
    off = ah.offbody_hair(full, sil, hair, margin_px=3)
    assert off[40, 200] and off[40, 215]          # outer whip, labelled body by the split
    assert not off[158, 60]                       # misfit arm off the model but not connected to her hair
    assert not off[100, 120] and not off[40, 120]  # body and head inside the model are never off-body
    assert not ah.offbody_hair(full, sil, None, 3).any()  # no hair mask: nothing


def test_hair_maps_extra_hair_and_whip_radius():
    full, sil, hair = _scene()
    off = ah.offbody_hair(full, sil, hair, margin_px=3)
    keep0, hb0 = ah.hair_maps(full, hair, None)
    keep1, hb1 = ah.hair_maps(full, hair, None, extra_hair=off)
    assert not hb0[40, 210] and hb1[40, 210] and keep1[40, 210] == 0
    # whip view: the exclusion around her hair reaches further into the body / head below and beside the whip
    keep3, _ = ah.hair_maps(full, hair, None, extra_hair=off, whip_mult=3.0)
    h = 210.0                                      # her mask height (rows 20..229)
    r1 = max(2.0, 0.006 * h) + max(2.0, 0.006 * h)  # normal ramp end (4 px); x3 for a whip: zero out to 6 px
    x = 140 - int(np.ceil(r1)) - 2                 # a head pixel 6 px left of the hair: outside the normal ramp
    assert full[45, x] and keep1[45, x] == 1.0 and keep3[45, x] == 0.0
    assert keep3[150, 120] == 1.0                  # the torso far from the hair keeps full weight
