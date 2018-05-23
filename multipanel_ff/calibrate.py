import numpy as np
import matplotlib as mpl
mpl.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from hexrd.gridutil import cellIndices
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd import instrument
from hexrd import imageseries

from skimage import io
from skimage import transform as tf
from skimage.exposure import equalize_hist, equalize_adapthist

Pimgs = imageseries.process.ProcessedImageSeries

tvec_DFLT = np.r_[0., 0., -1000.]
tilt_DFTL = np.zeros(3)

class InstrumentViewer(object):

    def __init__(self, instr, ims, planeData,
                 tilt=tilt_DFTL, tvec=tvec_DFLT, 
                 slider_delta=10., pixel_size=0.5):
        self.planeData = planeData
        self.instr = instr
        self._load_panels()
        self._load_images(ims)
        self.dplane = DisplayPlane(tvec=tvec)
        self.pixel_size = pixel_size
        self._make_dpanel()

        self._figure, self._axes = plt.subplots()
        plt.subplots_adjust(right=0.6)
        self._cax = None
        self._active_panel_id = None
        self.active_panel_mode = False
        self.image = None
        self.have_rings = False
        self.slider_delta = slider_delta
        self.set_interactors()
        self.show_image()
        plt.show()

    def _load_panels(self):
        self.panel_ids = self.instr._detectors.keys()
        self.panels = self.instr._detectors.values()

        # save original panel parameters for reset
        self.panel_vecs_orig = dict()
        for pid in self.panel_ids:
            p = self.instr._detectors[pid]
            self.panel_vecs_orig[pid] = (p.tvec, p.tilt)

    def _load_images(self, ims):
        # load images from imageseries
        # ... add processing here
        print "loading images"
        m = ims.metadata
        pids = m['panels']
        d = dict(zip(pids, range(len(pids))))

        if 'process' in m:
            pspec = m['process']
            ops = []
            for p in pspec:
                k = p.keys()[0]
                ops.append((k, p[k]))
            pims = Pimgs(ims, ops)
        else:
            pims = ims

        self.images = []
        for pid in self.panel_ids:
            self.images.append(pims[d[pid]])

    def _make_dpanel(self):
        self.dpanel_sizes = self.dplane.panel_size(self.instr)
        self.dpanel = self.dplane.display_panel(self.dpanel_sizes,
                                                self.pixel_size)

    def set_interactors(self):
        self._figure.canvas.mpl_connect('key_press_event', self.onkeypress)

        # sliders
        axcolor = 'lightgoldenrodyellow'

        # . translations
        self.tx_ax = plt.axes([0.65, 0.65, 0.30, 0.03], facecolor=axcolor)
        self.ty_ax = plt.axes([0.65, 0.60, 0.30, 0.03], facecolor=axcolor)
        self.tz_ax = plt.axes([0.65, 0.55, 0.30, 0.03], facecolor=axcolor)

        # . tilts
        self.gx_ax = plt.axes([0.65, 0.50, 0.30, 0.03], facecolor=axcolor)
        self.gy_ax = plt.axes([0.65, 0.45, 0.30, 0.03], facecolor=axcolor)
        self.gz_ax = plt.axes([0.65, 0.40, 0.30, 0.03], facecolor=axcolor)

        self._active_panel_id = self.panel_ids[0]
        panel = self.instr._detectors[self._active_panel_id]
        self._make_sliders(panel)

        # radio button (panel selector)
        rd_ax = plt.axes([0.65, 0.70, 0.30, 0.15], facecolor=axcolor)
        self.radio_panels = RadioButtons(rd_ax, self.panel_ids)
        self.radio_panels.on_clicked(self.on_change_panel)

    def _make_sliders(self, panel):
        """make sliders for given panel"""
        t = panel.tvec
        d = self.slider_delta

        g = np.degrees(panel.tilt)

        # translations
        self.tx_ax.clear()
        self.ty_ax.clear()
        self.tz_ax.clear()

        self.slider_tx = Slider(self.tx_ax, 't_x', t[0] - d, t[0] + d,
                                valinit=t[0])
        self.slider_ty = Slider(self.ty_ax, 't_y', t[1] - d, t[1] + d,
                                valinit=t[1])
        self.slider_tz = Slider(self.tz_ax, 't_z', t[2] - d, t[2] + d,
                                valinit=t[2])

        self.slider_tx.on_changed(self.update)
        self.slider_ty.on_changed(self.update)
        self.slider_tz.on_changed(self.update)

        # tilts
        self.gx_ax.clear()
        self.gy_ax.clear()
        self.gz_ax.clear()

        self.slider_gx = Slider(self.gx_ax, r'$\gamma_x$', g[0] - d, g[0] + d,
                                valinit=g[0])
        self.slider_gy = Slider(self.gy_ax, r'$\gamma_y$', g[1] - d, g[1] + d,
                                valinit=g[1])
        self.slider_gz = Slider(self.gz_ax, r'$\gamma_z$', g[2] - d, g[2] + d,
                                valinit=g[2])

        self.slider_gx.on_changed(self.update)
        self.slider_gy.on_changed(self.update)
        self.slider_gz.on_changed(self.update)


    # ========================= Properties
    @property
    def active_panel(self):
        return self.instr._detectors[self._active_panel_id]

    @property
    def instrument_output(self):
        tmpl = "new-instrument-%s.yml"
        if not hasattr(self, '_ouput_number'):
            self._ouput_number = 0
        else:
            self._ouput_number += 1

        return tmpl % self._ouput_number

    def onkeypress(self, event):
        #
        # r - reset panels
        # w - write instrument settings
        #
        print 'key press event: %s' % event.key
        if event.key in 'a':
            self.active_panel_mode = not self.active_panel_mode
            print "active panel mode is: %s" % self.active_panel_mode
        elif event.key in 'r':
            # Reset
            print "resetting panels"
            self.reset_panels()
        elif event.key in 'w':
            # Write config
            print "writing instrument config file"
            self.instr.write_config(self.instrument_output)
        elif event.key in 'i':
            ri = raw_input()
            print 'read: %s' % ri
        elif event.key in 'qQ':
            print "quitting"
            plt.close('all')
            return
        else:
            print("unrecognized key = %s\n" % event.key)

        self.show_image()

    def on_change_panel(self, id):
        self._active_panel_id = id
        panel = self.instr._detectors[id]
        self._make_sliders(panel)
        self.update(0)

    def reset_panels(self):
        for pid in self.panel_ids:
            p = self.instr._detectors[pid]
            tt = self.panel_vecs_orig[pid]
            p.tvec = tt[0]
            p.tilt = tt[1]

        self._make_sliders(self.active_panel)
        self.show_image()

    def update(self, val):
        panel = self.instr._detectors[self._active_panel_id]

        tvec = panel.tvec
        tvec[0] = self.slider_tx.val
        tvec[1] = self.slider_ty.val
        tvec[2] = self.slider_tz.val
        panel.tvec = tvec

        tilt = panel.tilt
        tilt[0] = np.radians(self.slider_gx.val)
        tilt[1] = np.radians(self.slider_gy.val)
        tilt[2] = np.radians(self.slider_gz.val)
        panel.tilt = tilt

        self.show_image()

    def show_image(self):
        # self._axes.clear()
        self._axes.set_title("Instrument")
        self.plot_dplane()
        self.addrings()

        # # colorbar
        # # cax = ax.imshow(np.log(1+ self._ims[k]))
        # del self._cax
        # self._cax = ax.imshow(self._ims[k])
        # if hasattr(self, 'cb'):
        #     self.cb.remove()
        # self.cb = fig.colorbar(self._cax)
        # self.cb.set_label('Some Units')
        plt.draw()

    def addrings(self):
        dp = self.dpanel
        if not self.have_rings:
            # generate and save rings
            ring_angs, ring_xys = dp.make_powder_rings(
                self.planeData, delta_eta=1)
            self.ring_data = []
            for ring in ring_xys:
                self.ring_data.append(dp.cartToPixel(ring))
            self.have_rings = True

            for pr in self.ring_data:
                self._axes.plot(pr[:, 1], pr[:, 0], 'c.', ms=2)
        #self._axes.set_xlim(-0.5, dp.cols-0.5)
        #self._axes.set_ylim(-0.5, dp.rows-0.5)


    def plot_dplane(self):
        dpanel = self.dpanel
        nrows_map = dpanel.rows
        ncols_map = dpanel.cols
        warped = np.zeros((nrows_map, ncols_map))
        for i in range(len(self.images)):
            detector_id = self.panel_ids[i]
            if self.active_panel_mode:
                if not detector_id == self._active_panel_id:
                    continue

            img = self.images[i]
            panel = self.instr._detectors[detector_id]

            # map corners
            corners = np.vstack(
                [panel.corner_ll,
                 panel.corner_lr,
                 panel.corner_ur,
                 panel.corner_ul,
                 ]
            )
            mp = panel.map_to_plane(corners, self.dplane.rmat, self.dplane.tvec)

            col_edges = dpanel.col_edge_vec
            row_edges = dpanel.row_edge_vec
            j_col = cellIndices(col_edges, mp[:, 0])
            i_row = cellIndices(row_edges, mp[:, 1])

            src = np.vstack([j_col, i_row]).T
            dst = panel.cartToPixel(corners, pixels=True)
            dst = dst[:, ::-1]

            tform3 = tf.ProjectiveTransform()
            tform3.estimate(src, dst)

            warped += tf.warp(img, tform3,
                              output_shape=(self.dpanel.rows,
                                            self.dpanel.cols))
        """
        IMAGE PLOTTING AND LIMIT CALCULATION
        """
        #img = equalize_adapthist(warped, clip_limit=0.05, nbins=2**16)
        #img = equalize_hist(warped, nbins=2**14)
        cmap = plt.cm.inferno
        cmap.set_under='b'
        img = warped
        vmin = np.percentile(img, 50)
        vmax = np.percentile(img, 99)
        if self.image is None:
            self.image = self._axes.imshow(
                    img, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    interpolation="none")
        else:
            self.image.set_data(img)
            self._figure.canvas.draw()
        self._axes.format_coord = self.format_coord

    def format_coord(self, j, i):
        """
        i, j are col, row
        """
        xy_data = self.dpanel.pixelToCart(np.vstack([i, j]).T)
        ang_data, gvec = self.dpanel.cart_to_angles(xy_data)
        tth = ang_data[:, 0]
        eta = ang_data[:, 1]
        dsp = 0.5 *self. planeData.wavelength / np.sin(0.5*tth)
        hkl = str(self.planeData.getHKLs(asStr=True, allHKLs=True, thisTTh=tth))
        return "x=%.2f, y=%.2f, d=%.3f tth=%.2f eta=%.2f HKLs=%s" \
          % (xy_data[0, 0], xy_data[0, 1], dsp, np.degrees(tth), np.degrees(eta), hkl)

    pass


class DisplayPlane(object):

    def __init__(self, tilt=tilt_DFTL, tvec=tvec_DFLT):
        self.tilt = tilt
        self.rmat = xfcapi.makeDetectorRotMat(self.tilt)
        self.tvec = tvec

    def panel_size(self, instr):
        """return bounding box of instrument panels in display plane"""
        xmin_i = ymin_i = np.inf
        xmax_i = ymax_i = -np.inf
        for detector_id in instr._detectors:
            panel = instr._detectors[detector_id]
            # find max extent
            corners = np.vstack(
                [panel.corner_ll,
                 panel.corner_lr,
                 panel.corner_ur,
                 panel.corner_ul,
                 ]
            )
            tmp = panel.map_to_plane(corners, self.rmat, self.tvec)
            xmin, xmax = np.sort(tmp[:, 0])[[0, -1]]
            ymin, ymax = np.sort(tmp[:, 1])[[0, -1]]

            xmin_i = min(xmin, xmin_i)
            ymin_i = min(ymin, ymin_i)
            xmax_i = max(xmax, xmax_i)
            ymax_i = max(ymax, ymax_i)
            pass

        del_x = 2*max(abs(xmin_i), abs(xmax_i))
        del_y = 2*max(abs(ymin_i), abs(ymax_i))

        return (del_x, del_y)

    def display_panel(self, sizes, mps):

        del_x = sizes[0]
        del_y = sizes[1]

        ncols_map = int(del_x/mps)
        nrows_map = int(del_y/mps)

        display_panel = instrument.PlanarDetector(
            rows=nrows_map, cols=ncols_map,
            pixel_size=(mps, mps),
            tvec=self.tvec, tilt=self.tilt)

        return display_panel
