import insightface
import roop.globals

FACE_ANALYSER = None
last_face = None

def get_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


#def get_face_single(img_data):
#    face = get_face_analyser().get(img_data)
#    try:
#        return sorted(face, key=lambda x: x.bbox[0])[0]
#    except IndexError:
#        return None


def get_face_idx(img_data, idx, allow_teleport=False):
    # this variable will keep track of the last faces position to prevent teleportation
    global last_face
    face = get_face_analyser().get(img_data)
    try:
        face_at_new_position = face[idx]
        if allow_teleport:
            return face[idx]
        else:
            if last_face is None:
                last_face = face_at_new_position
                return face_at_new_position
            else:
                # use 10% of the image width as a threshold for teleportation
                if abs(face_at_new_position.bbox[0] - last_face.bbox[0]) > (img_data.shape[1] * 0.1):
                    return last_face
                else:
                    last_face = face_at_new_position
                    return face_at_new_position
    except IndexError:
        return None


def get_face_many(img_data):
    try:
        return get_face_analyser().get(img_data)
    except IndexError:
        return None
