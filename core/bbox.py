

class BBox(object):
    def __init__(self, tlbr=None, tlwh=None, rel=False, class_name="", conf=0., imgsz=None):
        self.class_name = class_name
        self.confidence = conf
        self.img_h, self.img_w = imgsz[0], imgsz[1]
        self.tracking_id = -1

        if rel is False:     # 절대좌표
            if tlbr is not None:
                self.x1, self.y1, self.x2, self.y2 = map(int, tlbr)
                self.w = self.x2 - self.x1
                self.h = self.y2 - self.y1
            elif tlwh is not None:
                self.x1, self.y1, self.w, self.h = map(int, tlwh)
                self.x2 = self.x1 + self.w
                self.y2 = self.y1 + self.h
        else:       # 상대좌표
            if tlbr is not None:
                self.x1 = int(tlbr[0] * self.img_w)
                self.y1 = int(tlbr[1] * self.img_h)
                self.x2 = int(tlbr[2] * self.img_w)
                self.y2 = int(tlbr[3] * self.img_h)
                self.w = self.x2 - self.x1
                self.h = self.y2 - self.y1
            elif tlwh is not None:
                self.x1 = int(tlwh[0] * self.img_w)
                self.y1 = int(tlwh[1] * self.img_h)
                self.w = int(tlwh[2] * self.img_w)
                self.h = int(tlwh[3] * self.img_h)
                self.x2 = self.x1 + self.w
                self.y2 = self.y1 + self.h

    def __repr__(self):
        return f"{self.class_name}: {self.x1, self.y1, self.x2, self.y2}, conf:{self.confidence:.6f}"
