'''
    Apologies for the messy code - 'tis a hackathon afterall.
'''
from retinaface.model import Detector
import numpy as np
import traceback
import time
import cv2
import pylab
import os
import sys

def combine(left, right):
    """Stack images horizontally.
    """
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    hoff = left.shape[0]
    
    shape = list(left.shape)
    shape[0] = h
    shape[1] = w
    
    comb = np.zeros(tuple(shape),left.dtype)
    
    # left will be on left, aligned top, with right on right
    comb[:left.shape[0],:left.shape[1]] = left
    comb[:right.shape[0],left.shape[1]:] = right
    
    return comb   

class PulseMonitor(object):

    def __init__(self):
        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0

        self.face_detector = Detector()

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13

        self.idx = 1
        self.find_faces = True
        self.data_buffer, self.times = [], []
        self.plot_title = "Data Display"

    def plotXY(self, data, size=(280,640), margin=25, name="data", labels=[], skip=[],
            showmax=[], bg=None, label_ndigits=[], showmax_digits=[]):
        
        for x, y in data:
            if len(x) < 2 or len(y) < 2:
                return
        
        n_plots = len(data)
        w = float(size[1])
        h = size[0]/float(n_plots)
        
        z = np.zeros((size[0],size[1],3))
        
        # face overlay
        if isinstance(bg,np.ndarray):
            wd = int(bg.shape[1]/bg.shape[0]*h )
            bg = cv2.resize(bg,(wd,int(h)))
            if len(bg.shape) == 3:
                r = combine(bg[:,:,0],z[:,:,0])
                g = combine(bg[:,:,1],z[:,:,1])
                b = combine(bg[:,:,2],z[:,:,2])
            else:
                r = combine(bg,z[:,:,0])
                g = combine(bg,z[:,:,1])
                b = combine(bg,z[:,:,2])
            z = cv2.merge([r,g,b])[:,:-wd,]    
        
        i = 0
        P = []
        for x,y in data:
            x = np.array(x)
            y = -np.array(y)
            
            xx = (w-2*margin)*(x - x.min()) / (x.max() - x.min())+margin
            yy = (h-2*margin)*(y - y.min()) / (y.max() - y.min())+margin + i*h
            mx = max(yy)
            if labels:
                if labels[i]:
                    for ii in range(len(x)):
                        if ii%skip[i] == 0:
                            col = (255,255,255)
                            ss = '{0:.%sf}' % label_ndigits[i]
                            ss = ss.format(x[ii]) 
                            cv2.putText(z,ss,(int(xx[ii]),int((i+1)*h)),
                                        cv2.FONT_HERSHEY_PLAIN,1,col)           
            if showmax:
                if showmax[i]:
                    col = (0,255,0)    
                    ii = np.argmax(-y)
                    ss = '{0:.%sf} %s' % (showmax_digits[i], showmax[i])
                    ss = ss.format(x[ii]) 
                    #"%0.0f %s" % (x[ii], showmax[i])
                    cv2.putText(z,ss,(int(xx[ii]),int((yy[ii]))),
                                cv2.FONT_HERSHEY_PLAIN,2,col)
            
            try:
                pts = np.array([[x_, y_] for x_, y_ in zip(xx,yy)],np.int32)
                i+=1
                P.append(pts)
            except ValueError:
                pass #temporary
        """ 
        #Polylines seems to have some trouble rendering multiple polys for some people
        for p in P:
            cv2.polylines(z, [p], False, (255,255,255),1)
        """
        #hack-y alternative:
        for p in P:
            for i in range(len(p)-1):
                cv2.line(z,tuple(p[i]),tuple(p[i+1]), (255,255,255),1)    
        #cv2.imshow(name,z)
        #cv2.waitKey(1)
        return z


    def make_plot(self):
        graph = self.plotXY([[self.times, self.samples],
                [self.freqs, self.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=None) # bg=self.slices[0]
        
        if graph is not None:
            w, h, _ = np.shape(graph)
            graph = cv2.resize(graph, (int(h/2), int(w/2)))
            return graph

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])

        return (v1 + v2 + v3) / 3.

    def process_frame(self, frame):
        self.frame_in = frame
        self.run()
        return self.frame_out, self.bpm

    def run(self):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in, cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)

        detected = self.face_detector(self.frame_in).tolist()

        if len(detected) > 0:
            w = int(detected[0][2] - detected[0][0])
            h = int(detected[0][3] - detected[0][1])
            if self.shift(detected[0][:4]) > 4:
                self.face_rect = list(map(int, [detected[0][0], detected[0][1], w, h]))

            b = list(map(int, detected[0]))
            cv2.circle(self.frame_out, (b[5], b[6]), 1, (0, 0, 255), 4) # left eye
            cv2.circle(self.frame_out, (b[7], b[8]), 1, (0, 255, 255), 4) # right eye
            cv2.circle(self.frame_out, (b[9], b[10]), 1, (255, 0, 255), 4) # nose
            cv2.circle(self.frame_out, (b[11], b[12]), 1, (0, 255, 0), 4) # left mouth
            cv2.circle(self.frame_out, (b[13], b[14]), 1, (255, 0, 0), 4) # right mouth

        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(self.face_rect, col=(100, 255, 100))
        x, y, w, h = self.face_rect
        self.draw_rect(forehead1)
        x, y, w, h = forehead1

        if set(self.face_rect) == set([1, 1, 2, 2]):
            return

        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(forehead1)

        vals = self.get_subface_means(forehead1)

        self.data_buffer.append(vals)
        
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size
        
        processed = np.array(self.data_buffer)
        self.samples = processed
        if L > 10:
            self.output_dim = processed.shape[0]

            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times[:len(even_times)], processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 50) & (freqs < 180))
            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            self.bpm = self.freqs[idx2]
            self.idx += 1

            x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            r = alpha * self.frame_in[y:y + h, x:x + w, 0]
            g = alpha * self.frame_in[y:y + h, x:x + w, 1] + beta * self.gray[y:y + h, x:x + w]
            b = alpha * self.frame_in[y:y + h, x:x + w, 2]
            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,g,b])
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            col = (100, 255, 100)
            gap = (self.buffer_size - L) / self.fps

            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            else:
                text = "(estimate: %0.1f bpm)" % (self.bpm)
            tsize = 1
            cv2.putText(self.frame_out, text, (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, tsize, col)

            plot = self.make_plot()
            _h, _w, _c = np.shape(plot)
            self.frame_out[-_h:, :_w, :_c] = plot


if __name__ == "__main__":

    processor = PulseMonitor()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            output_frame = processor.process_frame(frame)
        except Exception as e:
            traceback.print_exc()

        cv2.imshow("Frame", output_frame)
        cv2.waitKey(1)
