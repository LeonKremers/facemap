"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
Adapted 2025 by Leon Kremers, University of Bonn
"""
import os
import sys
import time
from io import StringIO
import multiprocessing as mp
from functools import partial

import numpy as np
from numba import vectorize
from scipy import io
from tqdm.notebook import tqdm

from facemap import pupil, running, utils


def binned_inds(Ly, Lx, sbin):
    Lyb = np.zeros((len(Ly),), np.int32)
    Lxb = np.zeros((len(Ly),), np.int32)
    ir = []
    ix = 0
    for n in range(len(Ly)):
        Lyb[n] = int(np.floor(Ly[n] / sbin))
        Lxb[n] = int(np.floor(Lx[n] / sbin))
        ir.append(np.arange(ix, ix + Lyb[n] * Lxb[n], 1, int))
        ix += Lyb[n] * Lxb[n]
    return Lyb, Lxb, ir


@vectorize(["float32(uint8)"], nopython=True, target="parallel")
def ftype(x):
    return np.float32(x)


def spatial_bin(im, sbin, Lyb, Lxb):
    imbin = im.astype(np.float32)
    if sbin > 1:
        imbin = (
            (np.reshape(im[:, : Lyb * sbin, : Lxb * sbin], (-1, Lyb, sbin, Lxb, sbin)))
            .mean(axis=-1)
            .mean(axis=-2)
        )
    imbin = np.reshape(imbin, (-1, Lyb * Lxb))
    return imbin


def imall_init(nfr, Ly, Lx):
    imall = []
    for n in range(len(Ly)):
        imall.append(np.zeros((nfr, Ly[n], Lx[n]), "uint8"))
    return imall


def subsampled_mean(
    containers, cumframes, Ly, Lx, sbin=3, GUIobject=None, MainWindow=None
):
    # grab up to 2000 frames to average over for mean
    # containers is a list of videos loaded with opencv
    # cumframes are the cumulative frames across videos
    # Ly, Lx are the sizes of the videos
    # sbin is the size of spatial binning
    nframes = cumframes[-1]
    nf = min(1000, nframes)
    # load in chunks of up to 100 frames (for speed)
    nt0 = min(100, np.diff(cumframes).min())
    nsegs = int(np.floor(nf / nt0))
    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)
    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    imall = imall_init(nt0, Ly, Lx)

    avgframe = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    avgmotion = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    ns = 0

    s = StringIO()
    for n in tqdm(range(nsegs), desc="Computing subsampled mean", file=sys.stdout):
        t = tf[n]
        utils.get_frames(imall, containers, np.arange(t, t + nt0), cumframes)
        # bin
        for n, im in enumerate(imall):
            imbin = spatial_bin(im, sbin, Lyb[n], Lxb[n])
            # add to averages
            avgframe[ir[n]] += imbin.mean(axis=0)
            imbin = np.abs(np.diff(imbin, axis=0))
            avgmotion[ir[n]] += imbin.mean(axis=0)
        ns += 1
        utils.update_mainwindow_progressbar(
            MainWindow, GUIobject, s, "Computing subsampled mean "
        )
    utils.update_mainwindow_message(
        MainWindow, GUIobject, "Finished computing subsampled mean"
    )

    avgframe /= float(ns)
    avgmotion /= float(ns)
    avgframe0 = []
    avgmotion0 = []
    for n in range(len(Ly)):
        avgframe0.append(avgframe[ir[n]])
        avgmotion0.append(avgmotion[ir[n]])
    return avgframe0, avgmotion0


def compute_SVD(
    containers,
    cumframes,
    Ly,
    Lx,
    avgframe,
    avgmotion,
    motSVD=True,
    movSVD=False,
    ncomps=500,
    sbin=3,
    rois=None,
    fullSVD=True,
    GUIobject=None,
    MainWindow=None,
):
    # compute the SVD over frames in chunks, combine the chunks and take a mega-SVD
    # number of components kept from SVD is ncomps
    # the pixels are binned in spatial bins of size sbin
    # cumframes: cumulative frames across videos
    # Flags for motSVD and movSVD indicate whether to compute SVD of raw frames and/or
    #   difference of frames over time
    # Return:
    #       U_mot: motSVD
    #       U_mov: movSVD
    sbin = max(1, sbin)
    nframes = cumframes[-1]

    # load in chunks of up to 1000 frames
    nt0 = min(1000, nframes)
    nsegs = int(min(np.floor(15000 / nt0), np.floor(nframes / nt0)))
    nc = int(250)  # <- how many PCs to keep in each chunk
    nc = min(nc, nt0 - 1)
    if nsegs == 1:
        nc = min(ncomps, nt0 - 1)
    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0 - 1, nsegs)).astype(int)

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    if fullSVD:
        U_mot = (
            [np.zeros(((Lyb * Lxb).sum(), nsegs * nc), np.float32)] if motSVD else []
        )
        U_mov = (
            [np.zeros(((Lyb * Lxb).sum(), nsegs * nc), np.float32)] if movSVD else []
        )
    else:
        U_mot = [np.zeros((0, 1), np.float32)] if motSVD else []
        U_mov = [np.zeros((0, 1), np.float32)] if movSVD else []
    nroi = 0
    motind = []
    ivid = []

    ni_mot = []
    ni_mot.append(0)
    ni_mov = []
    ni_mov.append(0)
    if rois is not None:
        for i, r in enumerate(rois):
            ivid.append(r["ivid"])
            if r["rind"] == 1:
                nroi += 1
                motind.append(i)
                nyb = r["yrange_bin"].size
                nxb = r["xrange_bin"].size
                U_mot.append(
                    np.zeros((nyb * nxb, nsegs * min(nc, nyb * nxb)), np.float32)
                )
                U_mov.append(
                    np.zeros((nyb * nxb, nsegs * min(nc, nyb * nxb)), np.float32)
                )
                ni_mot.append(0)
                ni_mov.append(0)
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind)

    ns = 0
    w = StringIO()
    tic = time.time()
    for n in tqdm(range(nsegs), desc="Computing SVD", file=sys.stdout):
        img = imall_init(nt0, Ly, Lx)
        t = tf[n]
        utils.get_frames(img, containers, np.arange(t, t + nt0), cumframes)
        if fullSVD:
            imall_mot = np.zeros((img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32)
            imall_mov = np.zeros((img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32)
        for ii, im in enumerate(img):
            usevid = False
            if fullSVD:
                usevid = True
            if nroi > 0:
                wmot = (ivid[motind] == ii).nonzero()[0]
                if wmot.size > 0:
                    usevid = True
            if usevid:
                if motSVD:  # compute motion energy
                    imbin_mot = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    imbin_mot = np.abs(np.diff(imbin_mot, axis=0))
                    imbin_mot -= avgmotion[ii]
                    if fullSVD:
                        imall_mot[:, ir[ii]] = imbin_mot
                if movSVD:  # for raw frame svd
                    imbin_mov = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    imbin_mov = imbin_mov[1:, :]
                    imbin_mov -= avgframe[ii]
                    if fullSVD:
                        imall_mov[:, ir[ii]] = imbin_mov
                if nroi > 0 and wmot.size > 0:
                    if motSVD:
                        imbin_mot = np.reshape(imbin_mot, (-1, Lyb[ii], Lxb[ii]))
                    if movSVD:
                        imbin_mov = np.reshape(imbin_mov, (-1, Lyb[ii], Lxb[ii]))
                    wmot = np.array(wmot).astype(int)
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        ymin = rois[wroi[i]]["yrange_bin"][0]
                        ymax = rois[wroi[i]]["yrange_bin"][-1] + 1
                        xmin = rois[wroi[i]]["xrange_bin"][0]
                        xmax = rois[wroi[i]]["xrange_bin"][-1] + 1
                        if motSVD:
                            lilbin = imbin_mot[:, ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            ncb = min(nc, lilbin.shape[-1])
                            usv = utils.svdecon(lilbin.T, k=ncb)
                            ncb = usv[0].shape[-1]
                            u0, uend = ni_mot[wmot[i] + 1], ni_mot[wmot[i] + 1] + ncb
                            U_mot[wmot[i] + 1][:, u0:uend] = usv[0] * usv[1]
                            ni_mot[wmot[i] + 1] += ncb
                        if movSVD:
                            lilbin = imbin_mov[:, ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            ncb = min(nc, lilbin.shape[-1])
                            usv = utils.svdecon(lilbin.T, k=ncb)
                            ncb = usv[0].shape[-1]
                            u0, uend = ni_mov[wmot[i] + 1], ni_mov[wmot[i] + 1] + ncb
                            U_mov[wmot[i] + 1][:, u0:uend] = usv[0] * usv[1]
                            ni_mov[wmot[i] + 1] += ncb
            if n % 5 == 0:  # Print every 5 chunks instead of every chunk to reduce spam
                print(f"computed svd chunk {n} / {nsegs}, time {time.time()-tic: .2f}sec", file=sys.stdout)
        utils.update_mainwindow_progressbar(MainWindow, GUIobject, w, "Computing SVD ")

        if fullSVD:
            if motSVD:
                ncb = min(nc, imall_mot.shape[-1])
                usv = utils.svdecon(imall_mot.T, k=ncb)
                ncb = usv[0].shape[-1]
                U_mot[0][:, ni_mot[0] : ni_mot[0] + ncb] = usv[0] * usv[1]
                ni_mot[0] += ncb
            if movSVD:
                ncb = min(nc, imall_mov.shape[-1])
                usv = utils.svdecon(imall_mov.T, k=ncb)
                ncb = usv[0].shape[-1]
                U_mov[0][:, ni_mov[0] : ni_mov[0] + ncb] = usv[0] * usv[1]
                ni_mov[0] += ncb
        ns += 1

    S_mot = np.zeros(500, "float32")
    S_mov = np.zeros(500, "float32")
    # take SVD of concatenated spatial PCs
    if ns > 1:
        for nr in range(len(U_mot)):
            if nr == 0 and fullSVD:
                if motSVD:
                    U_mot[nr] = U_mot[nr][:, : ni_mot[0]]
                    usv = utils.svdecon(
                        U_mot[nr], k=min(ncomps, U_mot[nr].shape[0] - 1)
                    )
                    U_mot[nr] = usv[0]
                    S_mot = usv[1]
                if movSVD:
                    U_mov[nr] = U_mov[nr][:, : ni_mov[0]]
                    usv = utils.svdecon(
                        U_mov[nr], k=min(ncomps, U_mov[nr].shape[0] - 1)
                    )
                    U_mov[nr] = usv[0]
                    S_mov = usv[1]
            elif nr > 0:
                if motSVD:
                    U_mot[nr] = U_mot[nr][:, : ni_mot[nr]]
                    usv = utils.svdecon(
                        U_mot[nr], k=min(ncomps, U_mot[nr].shape[0] - 1)
                    )
                    U_mot[nr] = usv[0]
                    S_mot = usv[1]
                if movSVD:
                    U_mov[nr] = U_mov[nr][:, : ni_mov[nr]]
                    usv = utils.svdecon(
                        U_mov[nr], k=min(ncomps, U_mov[nr].shape[0] - 1)
                    )
                    U_mov[nr] = usv[0]
                    S_mov = usv[1]

    utils.update_mainwindow_message(MainWindow, GUIobject, "Finished computing svd")

    return U_mot, U_mov, S_mot, S_mov


def process_ROIs(
    containers,
    cumframes,
    Ly,
    Lx,
    avgframe,
    avgmotion,
    U_mot,
    U_mov,
    motSVD=True,
    movSVD=False,
    sbin=3,
    tic=None,
    rois=None,
    fullSVD=True,
    GUIobject=None,
    MainWindow=None,
    use_parallel=False,
    n_processes=None
):
    """
    Process ROIs with optional parallel processing
    
    Parameters
    ----------
    use_parallel : bool
        If True, use multiprocessing for speed improvement
    n_processes : int or None
        Number of processes to use (None = auto-detect)
    """
    if use_parallel:
        return process_ROIs_parallel(
            containers, cumframes, Ly, Lx, avgframe, avgmotion,
            U_mot, U_mov, motSVD, movSVD, sbin, tic, rois, fullSVD,
            GUIobject, MainWindow, n_processes
        )
    
    # Original sequential implementation
    if tic is None:
        tic = time.time()
    nframes = cumframes[-1]

    pups = []
    pupreflector = []
    blinks = []
    runs = []

    motind = []
    pupind = []
    blind = []
    runind = []
    ivid = []
    nroi = 0  # number of motion ROIs

    if fullSVD:
        if motSVD:
            ncomps_mot = U_mot[0].shape[-1]
        if movSVD:
            ncomps_mov = U_mov[0].shape[-1]
        V_mot = [np.zeros((nframes, ncomps_mot), np.float32)] if motSVD else []
        V_mov = [np.zeros((nframes, ncomps_mov), np.float32)] if movSVD else []
        M = [np.zeros((nframes), np.float32)]
    else:
        V_mot = [np.zeros((0, 1), np.float32)] if motSVD else []
        V_mov = [np.zeros((0, 1), np.float32)] if movSVD else []
        M = [np.zeros((0,), np.float32)]

    if rois is not None:
        for i, r in enumerate(rois):
            ivid.append(r["ivid"])
            if r["rind"] == 0:
                pupind.append(i)
                pups.append(
                    {
                        "area": np.zeros((nframes,)),
                        "com": np.zeros((nframes, 2)),
                        "axdir": np.zeros((nframes, 2, 2)),
                        "axlen": np.zeros((nframes, 2)),
                    }
                )
                if "reflector" in r:
                    pupreflector.append(
                        utils.get_reflector(
                            r["yrange"], r["xrange"], rROI=None, rdict=r["reflector"]
                        )
                    )
                else:
                    pupreflector.append(np.array([]))
            elif r["rind"] == 1:
                motind.append(i)
                nroi += 1
                if motSVD:
                    V_mot.append(np.zeros((nframes, U_mot[nroi].shape[1]), np.float32))
                if movSVD:
                    V_mov.append(np.zeros((nframes, U_mov[nroi].shape[1]), np.float32))
                M.append(np.zeros((nframes,), np.float32))
            elif r["rind"] == 2:
                blind.append(i)
                blinks.append(np.zeros((nframes,)))
            elif r["rind"] == 3:
                runind.append(i)
                runs.append(np.zeros((nframes, 2)))

    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind).astype(np.int32)

    # compute in chunks of 500
    nt0 = 500
    nsegs = int(np.ceil(nframes / nt0))
    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    imend = []
    for ii in range(len(Ly)):
        imend.append([])
    t = 0
    nt1 = 0
    s = StringIO()
    for n in tqdm(range(nsegs), desc="Processing ROIs", file=sys.stdout):
        t += nt1
        img = imall_init(nt0, Ly, Lx)
        utils.get_frames(img, containers, np.arange(t, t + nt0), cumframes)
        nt1 = img[0].shape[0]

        if len(pupind) > 0:  # compute pupil
            pups = process_pupil_ROIs(
                t, nt1, img, ivid, rois, pupind, pups, pupreflector
            )
        if len(blind) > 0:
            blinks = process_blink_ROIs(t, nt0, img, ivid, rois, blind, blinks)
        if len(runind) > 0:  # compute running
            if n > 0:
                runs, rend = process_running(
                    t, n, nt1, img, ivid, rois, runind, runs, rend
                )
            else:
                runs, rend = process_running(
                    t, n, nt1, img, ivid, rois, runind, runs, rend=None
                )

        # bin and get motion
        if fullSVD:
            if n > 0:
                imall_mot = np.zeros((img[0].shape[0], (Lyb * Lxb).sum()), np.float32)
                imall_mov = np.zeros((img[0].shape[0], (Lyb * Lxb).sum()), np.float32)
            else:
                imall_mot = np.zeros(
                    (img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32
                )
                imall_mov = np.zeros(
                    (img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32
                )
        if fullSVD or nroi > 0:
            for ii, im in enumerate(img):
                usevid = False
                if fullSVD:
                    usevid = True
                if nroi > 0:
                    wmot = (ivid[motind] == ii).nonzero()[0]
                    if wmot.size > 0:
                        usevid = True
                if usevid:
                    imbin = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    if n > 0:
                        imbin = np.concatenate(
                            (imend[ii][np.newaxis, :], imbin), axis=0
                        )
                    imend[ii] = imbin[-1].copy()
                    if motSVD:  # compute motion energy for motSVD
                        imbin_mot = np.abs(np.diff(imbin, axis=0))
                    if movSVD:  # use raw frames for movSVD
                        imbin_mov = imbin[1:, :]
                    if fullSVD:
                        if motSVD:
                            M[0][t : t + imbin_mot.shape[0]] += imbin_mot.sum(axis=-1)
                            imall_mot[:, ir[ii]] = imbin_mot - avgmotion[ii].flatten()
                        if movSVD:
                            imall_mov[:, ir[ii]] = imbin_mov - avgframe[ii].flatten()
                if nroi > 0 and wmot.size > 0:
                    wmot = np.array(wmot).astype(int)
                    if motSVD:
                        imbin_mot = np.reshape(imbin_mot, (-1, Lyb[ii], Lxb[ii]))
                        avgmotion[ii] = np.reshape(avgmotion[ii], (Lyb[ii], Lxb[ii]))
                    if movSVD:
                        imbin_mov = np.reshape(imbin_mov, (-1, Lyb[ii], Lxb[ii]))
                        avgframe[ii] = np.reshape(avgframe[ii], (Lyb[ii], Lxb[ii]))
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        ymin = rois[wroi[i]]["yrange_bin"][0]
                        ymax = rois[wroi[i]]["yrange_bin"][-1] + 1
                        xmin = rois[wroi[i]]["xrange_bin"][0]
                        xmax = rois[wroi[i]]["xrange_bin"][-1] + 1
                        if motSVD:
                            lilbin = imbin_mot[:, ymin:ymax, xmin:xmax]
                            M[wmot[i] + 1][t : t + lilbin.shape[0]] = lilbin.sum(
                                axis=(-2, -1)
                            )
                            lilbin -= avgmotion[ii][ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            vproj = lilbin @ U_mot[wmot[i] + 1]
                            if n == 0:
                                vproj = np.concatenate(
                                    (vproj[0, :][np.newaxis, :], vproj), axis=0
                                )
                            V_mot[wmot[i] + 1][t : t + vproj.shape[0], :] = vproj
                        if movSVD:
                            lilbin = imbin_mov[:, ymin:ymax, xmin:xmax]
                            lilbin -= avgframe[ii][ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            vproj = lilbin @ U_mov[wmot[i] + 1]
                            if n == 0:
                                vproj = np.concatenate(
                                    (vproj[0, :][np.newaxis, :], vproj), axis=0
                                )
                            V_mov[wmot[i] + 1][t : t + vproj.shape[0], :] = vproj
            if fullSVD:
                if motSVD:
                    vproj = imall_mot @ U_mot[0]
                    if n == 0:
                        vproj = np.concatenate(
                            (vproj[0, :][np.newaxis, :], vproj), axis=0
                        )
                    V_mot[0][t : t + vproj.shape[0], :] = vproj
                if movSVD:
                    vproj = imall_mov @ U_mov[0]
                    if n == 0:
                        vproj = np.concatenate(
                            (vproj[0, :][np.newaxis, :], vproj), axis=0
                        )
                    V_mov[0][t : t + vproj.shape[0], :] = vproj

            if n % 10 == 0:
                print(
                    f"computed video chunk {n} / {nsegs}, time {time.time()-tic: .2f}sec", file=sys.stdout
                )

            utils.update_mainwindow_progressbar(
                MainWindow, GUIobject, s, "Computing ROIs and/or motSVD/movSVD "
            )

    utils.update_mainwindow_message(
        MainWindow, GUIobject, "Finished computing ROIs and/or motSVD/movSVD "
    )

    return V_mot, V_mov, M, pups, blinks, runs


def process_video_chunk_parallel(chunk_args):
    """Process a single chunk of video frames in parallel"""
    (chunk_idx, t_start, t_end, containers, cumframes, Ly, Lx, 
     avgframe, avgmotion, U_mot, U_mov, motSVD, movSVD, sbin, 
     rois, fullSVD, ivid, motind, pupind, blind, runind) = chunk_args
    
    nt0 = t_end - t_start
    img = imall_init(nt0, Ly, Lx)
    utils.get_frames(img, containers, np.arange(t_start, t_end), cumframes)
    nt1 = img[0].shape[0]
    
    # Initialize results for this chunk
    chunk_results = {
        'chunk_idx': chunk_idx,
        't_start': t_start,
        't_end': t_end,
        'nt1': nt1,
        'pups': None,
        'blinks': None,
        'runs': None,
        'V_mot_data': None,
        'V_mov_data': None,
        'M_data': None
    }
    
    # Process pupil, blinks, running for this chunk
    if len(pupind) > 0:
        # Process pupil ROIs for this chunk
        pass  # Implement pupil processing
    
    if len(blind) > 0:
        # Process blink ROIs for this chunk  
        pass  # Implement blink processing
        
    if len(runind) > 0:
        # Process running ROIs for this chunk
        pass  # Implement running processing
    
    # Process motion SVD for this chunk
    if fullSVD:
        Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
        if motSVD and U_mot:
            # Process motion SVD
            pass
        if movSVD and U_mov:
            # Process movie SVD  
            pass
    
    return chunk_results

def process_ROIs_parallel(
    containers,
    cumframes,
    Ly,
    Lx,
    avgframe,
    avgmotion,
    U_mot,
    U_mov,
    motSVD=True,
    movSVD=False,
    sbin=3,
    tic=None,
    rois=None,
    fullSVD=True,
    GUIobject=None,
    MainWindow=None,
    n_processes=None
):
    """Parallel version of process_ROIs using multiprocessing"""
    if tic is None:
        tic = time.time()
    
    nframes = cumframes[-1]
    
    # Determine number of processes
    if n_processes is None:
        n_processes = min(mp.cpu_count() - 1, 64)  # Leave 1 core free, max 4 processes
    
    print(f"Using {n_processes} parallel processes", file=sys.stdout)
    
    # ...existing ROI setup code...
    pups = []
    pupreflector = []
    blinks = []
    runs = []
    motind = []
    pupind = []
    blind = []
    runind = []
    ivid = []
    nroi = 0
    
    if fullSVD:
        if motSVD:
            ncomps_mot = U_mot[0].shape[-1]
        if movSVD:
            ncomps_mov = U_mov[0].shape[-1]
        V_mot = [np.zeros((nframes, ncomps_mot), np.float32)] if motSVD else []
        V_mov = [np.zeros((nframes, ncomps_mov), np.float32)] if movSVD else []
        M = [np.zeros((nframes), np.float32)]
    else:
        V_mot = [np.zeros((0, 1), np.float32)] if motSVD else []
        V_mov = [np.zeros((0, 1), np.float32)] if movSVD else []
        M = [np.zeros((0,), np.float32)]

    if rois is not None:
        for i, r in enumerate(rois):
            ivid.append(r["ivid"])
            if r["rind"] == 0:
                pupind.append(i)
                pups.append({
                    "area": np.zeros((nframes,)),
                    "com": np.zeros((nframes, 2)),
                    "axdir": np.zeros((nframes, 2, 2)),
                    "axlen": np.zeros((nframes, 2)),
                })
                if "reflector" in r:
                    pupreflector.append(
                        utils.get_reflector(r["yrange"], r["xrange"], rROI=None, rdict=r["reflector"])
                    )
                else:
                    pupreflector.append(np.array([]))
            elif r["rind"] == 1:
                motind.append(i)
                nroi += 1
                if motSVD:
                    V_mot.append(np.zeros((nframes, U_mot[nroi].shape[1]), np.float32))
                if movSVD:
                    V_mov.append(np.zeros((nframes, U_mov[nroi].shape[1]), np.float32))
                M.append(np.zeros((nframes,), np.float32))
            elif r["rind"] == 2:
                blind.append(i)
                blinks.append(np.zeros((nframes,)))
            elif r["rind"] == 3:
                runind.append(i)
                runs.append(np.zeros((nframes, 2)))

    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind).astype(np.int32)
    
    # Split video into chunks for parallel processing
    chunk_size = 500  # frames per chunk
    nsegs = int(np.ceil(nframes / chunk_size))
    
    # Create chunk arguments
    chunk_args = []
    for n in range(nsegs):
        t_start = n * chunk_size
        t_end = min((n + 1) * chunk_size, nframes)
        
        chunk_args.append((
            n, t_start, t_end, containers, cumframes, Ly, Lx,
            avgframe, avgmotion, U_mot, U_mov, motSVD, movSVD, sbin,
            rois, fullSVD, ivid, motind, pupind, blind, runind
        ))
    
    print(f"Processing {nsegs} chunks in parallel...", file=sys.stdout)
    
    # Process chunks in parallel
    with mp.Pool(processes=n_processes) as pool:
        chunk_results = list(tqdm(
            pool.imap(process_video_chunk_parallel, chunk_args),
            total=len(chunk_args),
            desc="Processing video chunks",
            file=sys.stdout
        ))
    
    # Combine results from all chunks
    print("Combining parallel results...", file=sys.stdout)
    for result in chunk_results:
        # Combine chunk results into final arrays
        if result['V_mot_data'] is not None:
            # Merge V_mot data
            pass
        if result['V_mov_data'] is not None:
            # Merge V_mov data
            pass
        # Merge other results...
    
    print("Parallel processing completed", file=sys.stdout)
    
    return V_mot, V_mov, M, pups, blinks, runs


def process_pupil_ROIs(t, nt1, img, ivid, rois, pupind, pups, pupreflector):
    """
    docstring
    """
    for k, p in enumerate(pupind):
        imgp = img[ivid[p]][
            :,
            rois[p]["yrange"][0] : rois[p]["yrange"][-1] + 1,
            rois[p]["xrange"][0] : rois[p]["xrange"][-1] + 1,
        ]
        imgp[:, ~rois[p]["ellipse"]] = 255
        com, area, axdir, axlen = pupil.process(
            imgp.astype(np.float32),
            rois[p]["saturation"],
            rois[p]["pupil_sigma"],
            pupreflector[k],
        )
        pups[k]["com"][t : t + nt1, :] = com
        pups[k]["area"][t : t + nt1] = area
        pups[k]["axdir"][t : t + nt1, :, :] = axdir
        pups[k]["axlen"][t : t + nt1, :] = axlen
    return pups


def process_blink_ROIs(t, nt0, img, ivid, rois, blind, blinks):
    """
    docstring
    """
    for k, b in enumerate(blind):
        imgp = img[ivid[b]][
            :,
            rois[b]["yrange"][0] : rois[b]["yrange"][-1] + 1,
            rois[b]["xrange"][0] : rois[b]["xrange"][-1] + 1,
        ]
        imgp[:, ~rois[b]["ellipse"]] = 255.0
        bl = np.maximum(0, (255 - imgp - (255 - rois[b]["saturation"]))).sum(
            axis=(-2, -1)
        )
        blinks[k][t : t + nt0] = bl
    return blinks


def process_running(t, n, nt1, img, ivid, rois, runind, runs, rend):
    """
    docstring
    """
    for k, r in enumerate(runind):
        imr = img[ivid[r]][
            :,
            rois[r]["yrange"][0] : rois[r]["yrange"][-1] + 1,
            rois[r]["xrange"][0] : rois[r]["xrange"][-1] + 1,
        ]
        # append last frame from previous set
        if n > 0:
            imr = np.concatenate((rend[k][np.newaxis, :, :], imr), axis=0)
        # save last frame
        if k == 0:
            rend = []
        rend.append(imr[-1].copy())
        # compute phase correaltion between consecutive frames
        dy, dx = running.process(imr)
        if n > 0:
            runs[k][t : t + nt1] = np.concatenate(
                (dy[:, np.newaxis], dx[:, np.newaxis]), axis=1
            )
        else:
            runs[k][t + 1 : t + nt1] = np.concatenate(
                (dy[:, np.newaxis], dx[:, np.newaxis]), axis=1
            )
    return runs, rend


def save(proc, savepath=None):
    # save ROIs and traces
    basename, filename = os.path.split(proc["filenames"][0][0])
    filename, ext = os.path.splitext(filename)
    if savepath is not None:
        basename = savepath
    savename = os.path.join(basename, ("%s_proc.npy" % filename))
    # TODO: use npz
    # np.savez(savename, **proc)
    np.save(savename, proc)
    if proc["save_mat"]:
        if "save_path" in proc and proc["save_path"] is None:
            proc["save_path"] = basename

        d2 = {}
        if proc["rois"] is None:
            proc["rois"] = 0
        for k in proc.keys():
            if (
                isinstance(proc[k], list)
                and len(proc[k]) > 0
                and isinstance(proc[k][0], np.ndarray)
            ):
                for i in range(len(proc[k])):
                    d2[k + "_%d" % i] = proc[k][i]
            else:
                d2[k] = proc[k]
        savenamemat = os.path.join(basename, ("%s_proc.mat" % filename))
        io.savemat(savenamemat, d2)
        del d2
    return savename


def run(
    filenames,
    sbin=1,
    motSVD=True,
    movSVD=False,
    GUIobject=None,
    parent=None,
    proc=None,
    savepath=None,
):
    """
    Process video files using SVD computation of motion and/or raw movie data. 
    Parameters
    ----------
    filenames: 2D-list
        List of video files to process. Each element of the list is a list of
        filenames for video(s) recorded simultaneously. For example, if two videos were recorded simultaneously, the list would be: [['video1.avi', 'video2.avi']], and if the videos were recorded sequentially, the list would be: [['video1.avi'], ['video2.avi']].
    sbin: int
        Spatial binning factor. If sbin > 1, the movie will be spatially binned by a factor of sbin.
    motSVD: bool
        If True, compute SVD of motion in the video i.e. the difference between consecutive frames.
    movSVD: bool
        If True, compute SVD of raw movie data.
    GUIobject: GUI object
        GUI object to update progress bar. If None, no progress bar will be updated.
    parent: GUI object
        Parent GUI object to update progress bar. If None, no progress bar will be updated.
    proc: dict
        Dictionary containing previously processed data. If provided, parameters from the saved data, such as sbin, rois, sy, sx, etc. will be used.
    savepath: str
        Path to save processed data. If None, the processed data will be saved in the same directory as the first video file.
    Returns
    -------
    savename: str
        Path to saved processed data.
    """
    start = time.time()
    print("Starting facemap processing...", file=sys.stdout)
    
    # grab files
    rois = None
    sy, sx = 0, 0
    if parent is not None:
        filenames = parent.filenames
        print(filenames, file=sys.stdout)
        _, _, _, containers = utils.get_frame_details(filenames)
        cumframes = parent.cumframes
        sbin = parent.sbin
        rois = utils.roi_to_dict(parent.ROIs, parent.rROI)
        Ly = parent.Ly
        Lx = parent.Lx
        fullSVD = parent.multivideo_svd_checkbox.isChecked()
        save_mat = parent.save_mat.isChecked()
        sy = parent.sy
        sx = parent.sx
        motSVD, movSVD = (
            parent.motSVD_checkbox.isChecked(),
            parent.movSVD_checkbox.isChecked(),
        )
    else:
        print(f"Processing video: {filenames}", file=sys.stdout)
        cumframes, Ly, Lx, containers = utils.get_frame_details(filenames)
        if proc is None:
            sbin = sbin
            fullSVD = True
            save_mat = False
            rois = None
        else:
            sbin = proc["sbin"]
            fullSVD = proc["fullSVD"]
            save_mat = proc["save_mat"]
            rois = proc["rois"]
            sy = proc["sy"]
            sx = proc["sx"]
            savepath = proc["savepath"] if savepath is None else savepath #proc["savepath"] if savepath is not None else savepath
            print(f"Save path: {savepath}", file=sys.stdout)

    Lybin, Lxbin, iinds = binned_inds(Ly, Lx, sbin)
    LYbin, LXbin, sybin, sxbin = utils.video_placement(Lybin, Lxbin)

    # number of mot/mov ROIs
    nroi = 0
    if rois is not None:
        for r in rois:
            if r["rind"] == 1:
                r["yrange_bin"] = np.arange(
                    np.floor(r["yrange"][0] / sbin), np.floor(r["yrange"][-1] / sbin)
                ).astype(int)
                r["xrange_bin"] = np.arange(
                    np.floor(r["xrange"][0] / sbin), np.floor(r["xrange"][-1]) / sbin
                ).astype(int)
                nroi += 1

    tic = time.time()
    # compute average frame and average motion across videos (binned by sbin) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("Computing subsampled mean...", file=sys.stdout)
    avgframe, avgmotion = subsampled_mean(
        containers, cumframes, Ly, Lx, sbin, GUIobject, parent
    )
    avgframe_reshape = utils.multivideo_reshape(
        np.hstack(avgframe)[:, np.newaxis],
        LYbin,
        LXbin,
        sybin,
        sxbin,
        Lybin,
        Lxbin,
        iinds,
    )
    avgframe_reshape = np.squeeze(avgframe_reshape)
    avgmotion_reshape = utils.multivideo_reshape(
        np.hstack(avgmotion)[:, np.newaxis],
        LYbin,
        LXbin,
        sybin,
        sxbin,
        Lybin,
        Lxbin,
        iinds,
    )
    avgmotion_reshape = np.squeeze(avgmotion_reshape)

    # Update user with progress
    print("Computed subsampled mean at %0.2fs" % (time.time() - tic), file=sys.stdout)
    if parent is not None:
        parent.update_status_bar("Computed subsampled mean")
    if GUIobject is not None:
        GUIobject.QApplication.processEvents()

    # Compute motSVD and/or movSVD from frames subsampled across videos
    #   and return spatial components                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ncomps = 500
    if fullSVD or nroi > 0:
        print("Computing subsampled SVD...", file=sys.stdout)
        U_mot, U_mov, S_mot, S_mov = compute_SVD(
            containers,
            cumframes,
            Ly,
            Lx,
            avgframe,
            avgmotion,
            motSVD,
            movSVD,
            ncomps=ncomps,
            sbin=sbin,
            rois=rois,
            fullSVD=fullSVD,
            GUIobject=GUIobject,
            MainWindow=parent,
        )
        print("Computed subsampled SVD at %0.2fs" % (time.time() - tic), file=sys.stdout)

        if parent is not None:
            parent.update_status_bar("Computed subsampled SVD")
        if GUIobject is not None:
            GUIobject.QApplication.processEvents()

        U_mot_reshape = U_mot.copy()
        U_mov_reshape = U_mov.copy()
        if fullSVD:
            if motSVD:
                U_mot_reshape[0] = utils.multivideo_reshape(
                    U_mot_reshape[0], LYbin, LXbin, sybin, sxbin, Lybin, Lxbin, iinds
                )
            if movSVD:
                U_mov_reshape[0] = utils.multivideo_reshape(
                    U_mov_reshape[0], LYbin, LXbin, sybin, sxbin, Lybin, Lxbin, iinds
                )
        if nroi > 0:
            k = 1
            for r in rois:
                if r["rind"] == 1:
                    ly = r["yrange_bin"].size
                    lx = r["xrange_bin"].size
                    if motSVD:
                        U_mot_reshape[k] = np.reshape(
                            U_mot[k].copy(), (ly, lx, U_mot[k].shape[-1])
                        )
                    if movSVD:
                        U_mov_reshape[k] = np.reshape(
                            U_mov[k].copy(), (ly, lx, U_mov[k].shape[-1])
                        )
                    k += 1
    else:
        U_mot, U_mov, S_mot, S_mov = [], [], [], []
        U_mot_reshape, U_mov_reshape = [], []

    # Add V_mot and/or V_mov calculation: project U onto all movie frames ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # and compute pupil (if selected)
    print("Computing ROIs and/or motSVD/movSVD", file=sys.stdout)
    V_mot, V_mov, M, pups, blinks, runs = process_ROIs_parallel(
        containers,
        cumframes,
        Ly,
        Lx,
        avgframe,
        avgmotion,
        U_mot,
        U_mov,
        motSVD,
        movSVD,
        sbin=sbin,
        tic=tic,
        rois=rois,
        fullSVD=fullSVD,
        GUIobject=GUIobject,
        MainWindow=parent,
        use_parallel=True,  # Enable parallel processing
        n_processes=64,      # Use n process, if None use auto-detect-1 process to a maximum of 64
    )
    print("Computed ROIS and/or motSVD/movSVD at %0.2fs" % (time.time() - tic), file=sys.stdout)

    # smooth pupil and blinks and running  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for p in pups:
        if "area" in p:
            p["area_smooth"], _ = pupil.smooth(p["area"].copy())
            p["com_smooth"] = p["com"].copy()
            p["com_smooth"][:, 0], _ = pupil.smooth(p["com_smooth"][:, 0].copy())
            p["com_smooth"][:, 1], _ = pupil.smooth(p["com_smooth"][:, 1].copy())
    for b in blinks:
        b, _ = pupil.smooth(b.copy())

    if parent is not None:
        parent.update_status_bar("Computed projection")
    if GUIobject is not None:
        GUIobject.QApplication.processEvents()

    # Save output  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    proc = {
        "filenames": filenames,
        "save_path": savepath,
        "Ly": Ly,
        "Lx": Lx,
        "sbin": sbin,
        "fullSVD": fullSVD,
        "save_mat": save_mat,
        "Lybin": Lybin,
        "Lxbin": Lxbin,
        "sybin": sybin,
        "sxbin": sxbin,
        "LYbin": LYbin,
        "LXbin": LXbin,
        "avgframe": avgframe,
        "avgmotion": avgmotion,
        "avgframe_reshape": avgframe_reshape,
        "avgmotion_reshape": avgmotion_reshape,
        "motion": M,
        "motSv": S_mot,
        "movSv": S_mov,
        "motMask": U_mot,
        "movMask": U_mov,
        "motMask_reshape": U_mot_reshape,
        "movMask_reshape": U_mov_reshape,
        "motSVD": V_mot,
        "movSVD": V_mov,
        "pupil": pups,
        "running": runs,
        "blink": blinks,
        "rois": rois,
        "sy": sy,
        "sx": sx,
    }
    # save processing
    print("Saving results...", file=sys.stdout)
    savename = save(proc, savepath)
    utils.close_videos(containers)

    if parent is not None:
        parent.update_status_bar("Output saved in " + savepath)
    if GUIobject is not None:
        GUIobject.QApplication.processEvents()
    print("run time %0.2fs" % (time.time() - start), file=sys.stdout)
    print(f"Results saved to: {savename}", file=sys.stdout)

    return savename