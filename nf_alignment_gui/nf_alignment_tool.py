import numpy as np
import matplotlib as mpl
mpl.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from hexrd.gridutil import cellIndices, make_tolerance_grid
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd import instrument
from hexrd import imageseries

from skimage import io
from skimage import transform as tf
from skimage.exposure import equalize_adapthist

Pimgs = imageseries.process.ProcessedImageSeries
Oimgs = imageseries.omega.OmegaImageSeries

USE_NUMBA = True
save_max_frames = True

# FIXME: get rid of dpanel stuff here; not relevant for NF
tvec_DFLT = np.r_[0., 1.4, -5.5]
tilt_DFLT = np.zeros(3)
'''
tvec_DFLT = np.r_[0., 0., -2000.]
tilt_DFLT = np.zeros(3)
'''
class InstrumentViewer(object):
    """
    cobbled from multipanel version

    TODO:
        * get rid of dpanel in here; not needed
        * port to Qt5Agg
    """
    def __init__(self, instr, ims, planeData, grain_param_list,
                 grain_ids=None, make_images=False,
                 ome_tol=1.0, ome_ranges=[(-np.pi, np.pi),],
                 tilt=tilt_DFLT, tvec=tvec_DFLT,
                 ):

        # instrument stuff
        self.instr = instr
        self.chi = instr.chi
        self.tvec = instr.tvec

        pixel_sizes = []
        for k, v in instr.detectors.iteritems():
            pixel_sizes.append(
                [v.pixel_size_row, v.pixel_size_col]
            )
        self.pixel_size = 2*np.min(pixel_sizes)

        # deal with grain parameters
        grain_param_list = np.atleast_2d(grain_param_list)
        if grain_ids is None:
            self._grain_ids = range(len(grain_param_list))
        else:
            self._grain_ids = grain_ids
        self.gpl = np.atleast_2d(grain_param_list[self._grain_ids])

        self._make_images = make_images
        self._ome_tol = ome_tol
        self.ome_ranges = ome_ranges

        self.planeData = planeData

        self._load_panels()
        self._load_images(ims)
        self.dplane = DisplayPlane(tilt=tilt, tvec=tvec)

        # FIXME: get rid of dpanel stuff here; not relevant for NF
        self._make_dpanel()

        self._figure, self._axes = plt.subplots()
        plt.subplots_adjust(right=0.6)
        self._cax = None
        self._active_panel_id = None
        self.active_panel_mode = False
        self.image = None
        self.have_overlay = False
        self.tvec_slider_delta = 0.5
        self.tilt_slider_delta = 3.0
        self.set_interactors()
        self.show_image()
        plt.show()

    def _load_panels(self):
        self.panel_ids = self.instr._detectors.keys()
        self.panels = self.instr._detectors.values()

        # save original panel parameters for reset
        self.panel_vecs_orig = dict()
        for panel_id in self.panel_ids:
            p = self.instr._detectors[panel_id]
            self.panel_vecs_orig[panel_id] = (p.tvec, p.tilt)

    def _load_images(self, ims):
        # load images from imageseries
        # ... add processing here
        print "loading images"
        if self._make_images:
            assert isinstance(ims, dict), \
                "To make images, ims input must be a dictionary"
            print "making max frames to spec..."
            max_frames = []
            for panel_id in self.panel_ids:
                panel = self.instr.detectors[panel_id]

                oims = Oimgs(ims[panel_id])  # now have OmegaImageSeries
                del_ome = oims.omega[0, 1] - oims.omega[0, 0]  # degrees
                simd = panel.simulate_rotation_series(
                    self.planeData, self.gpl,
                    self.ome_ranges, chi=self.chi, tVec_s=self.tvec
                    )
                pred_omes = np.degrees(
                    np.vstack(simd[2])[:, 2]
                )  # in DEGREES

                ndiv, tol_grid = make_tolerance_grid(
                    del_ome, self._ome_tol, 1,
                    adjust_window=True)
                frame_indices = []
                for ome in pred_omes:
                    expanded_omes = ome + tol_grid
                    fidxs = oims.omegarange_to_frames(
                        expanded_omes[0], expanded_omes[-1])
                    if len(fidxs) > 0:
                        frame_indices += fidxs
                if len(frame_indices) == 0:
                    raise RuntimeError, \
                        "no omegas in speficied imageseries range(s)"
                max_frames.append(
                    np.max(np.array([oims[k] for k in frame_indices]), axis=0)
                )
                pass  # closes loop on panels
            # max array-based ims of max frames
            # NOTE: this assumes that the frames are all the same, which
            # is ok for NF detector at different L distances...
            ims = imageseries.open(
                None, 'array',
                data=np.array(max_frames),
                meta=dict(panels=self.panel_ids))
            gid_str = ''
            for s in ['%s-' %i for i in self._grain_ids]:
                gid_str += s
            if save_max_frames:
                imageseries.write(
                    ims,
                    'imageseries-max_grains_%s.h5' %gid_str[:-1],
                    'hdf5', path='data')
            pass  # closes conditional on make_images
        m = ims.metadata
        panel_ids = m['panels']
        d = dict(zip(panel_ids, range(len(panel_ids))))

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
        for panel_id in self.panel_ids:
            self.images.append(pims[d[panel_id]])

    # FIXME: get rid of dpanel stuff here; not relevant for NF
    def _make_dpanel(self):
        self.dpanel_sizes = self.dplane.panel_size(self.instr)
        self.dpanel = self.dplane.display_panel(self.dpanel_sizes,
                                                self.pixel_size)

    def set_interactors(self):
        self._figure.canvas.mpl_connect('key_press_event', self.onkeypress)

        # sliders
        axcolor = 'lightgoldenrodyellow'

        # . translations
        self.tx_ax = plt.axes([0.65, 0.65, 0.30, 0.03], axisbg=axcolor)
        self.ty_ax = plt.axes([0.65, 0.60, 0.30, 0.03], axisbg=axcolor)
        self.tz_ax = plt.axes([0.65, 0.55, 0.30, 0.03], axisbg=axcolor)

        # . tilts
        self.gx_ax = plt.axes([0.65, 0.50, 0.30, 0.03], axisbg=axcolor)
        self.gy_ax = plt.axes([0.65, 0.45, 0.30, 0.03], axisbg=axcolor)
        self.gz_ax = plt.axes([0.65, 0.40, 0.30, 0.03], axisbg=axcolor)

        self._active_panel_id = self.panel_ids[0]
        panel = self.instr._detectors[self._active_panel_id]
        self._make_sliders(panel)

        # radio button (panel selector)
        rd_ax = plt.axes([0.65, 0.70, 0.30, 0.15], axisbg=axcolor)
        self.radio_panels = RadioButtons(rd_ax, self.panel_ids)
        self.radio_panels.on_clicked(self.on_change_panel)

    def _make_sliders(self, panel):
        """make sliders for given panel"""
        t = panel.tvec
        del_tv = self.tvec_slider_delta
        del_tl = self.tilt_slider_delta

        g = np.degrees(panel.tilt)

        # translations
        self.tx_ax.clear()
        self.ty_ax.clear()
        self.tz_ax.clear()

        self.slider_tx = Slider(self.tx_ax, 't_x',
                                t[0] - del_tv, t[0] + del_tv,
                                valinit=t[0])
        self.slider_ty = Slider(self.ty_ax, 't_y',
                                t[1] - del_tv, t[1] + del_tv,
                                valinit=t[1])
        self.slider_tz = Slider(self.tz_ax, 't_z',
                                t[2] - del_tv, t[2] + del_tv,
                                valinit=t[2])

        self.slider_tx.on_changed(self.update)
        self.slider_ty.on_changed(self.update)
        self.slider_tz.on_changed(self.update)

        # tilts
        self.gx_ax.clear()
        self.gy_ax.clear()
        self.gz_ax.clear()

        self.slider_gx = Slider(self.gx_ax, r'$\gamma_x$',
                                g[0] - del_tl, g[0] + del_tl,
                                valinit=g[0])
        self.slider_gy = Slider(self.gy_ax, r'$\gamma_y$',
                                g[1] - del_tl, g[1] + del_tl,
                                valinit=g[1])
        self.slider_gz = Slider(self.gz_ax, r'$\gamma_z$',
                                g[2] - del_tl, g[2] + del_tl,
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
        for panel_id in self.panel_ids:
            p = self.instr._detectors[panel_id]
            tt = self.panel_vecs_orig[panel_id]
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

        # redo simulation
        valid_ids, valid_hkls, valid_angs, valid_xys, ang_pixel_size = \
          panel.simulate_rotation_series(
              self.planeData, self.gpl,
              self.ome_ranges,
              chi=self.chi)

        # generate and save rings
        self.sim_data = []
        for xy in valid_xys:
            self.sim_data.append(
                panel.cartToPixel(xy)
            )
        ijs = np.vstack(self.sim_data)
        self.overlay.set_xdata(ijs[:, 1])
        self.overlay.set_ydata(ijs[:, 0])

        self.show_image()

    def show_image(self):
        # self._axes.clear()
        self._axes.set_title("Instrument")
        # self.plot_dplane()
        self.plot_dummy()
        self.addpoints()
        plt.draw()

    def addpoints(self):
        # FIXME: get rid of dpanel stuff here; not relevant for NF.
        # must change initialization of simulation to ref active panel
        if not self.have_overlay:
            dp = self.dpanel
            valid_ids, valid_hkls, valid_angs, valid_xys, ang_pixel_size = \
              dp.simulate_rotation_series(
                  self.planeData, self.gpl,
                  self.ome_ranges,
                  chi=self.chi)

            # generate and save rings
            self.sim_data = []
            for xy in valid_xys:
                self.sim_data.append(
                    dp.cartToPixel(xy)
                )
            ijs = np.vstack(self.sim_data)
            #import pdb; pdb.set_trace()
            self.overlay, = self._axes.plot(ijs[:, 1], ijs[:, 0], 'cs', ms=4)
            self.have_overlay = True

    # FIXME: get rid of dpanel stuff here; not relevant for NF.
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
        img = equalize_adapthist(warped, clip_limit=0.1, nbins=2**16)
        if self.image is None:
            self.image = self._axes.imshow(
                    img, cmap=plt.cm.bone,
                    vmax=None,
                    interpolation="none")
        else:
            self.image.set_data(img)
            self._figure.canvas.draw()
        # self._axes.format_coord = self.format_coord

    def plot_dummy(self):
        for i in range(len(self.images)):
            detector_id = self.panel_ids[i]
            if self.active_panel_mode:
                if not detector_id == self._active_panel_id:
                    continue

            img = equalize_adapthist(self.images[i], clip_limit=0.05, nbins=2**16)
            panel = self.instr._detectors[detector_id]
        if self.image is None:
            self.image = self._axes.imshow(
                    img, cmap=plt.cm.bone,
                    vmax=None,
                    interpolation="none")
        else:
            self.image.set_data(img)
            self._figure.canvas.draw()

    # TODO: get rid of dpanel stuff here; not relevant for NF.
    '''
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
    '''
    pass

class DisplayPlane(object):

    def __init__(self, tilt=tilt_DFLT, tvec=tvec_DFLT):
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
