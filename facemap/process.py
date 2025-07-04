"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
Adapted 2025 by Leon Kremers, University of Bonn
"""

import os
import sys
import time
from io import StringIO
import numpy as np
from numba import vectorize
from scipy import io
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from facemap import pupil, running, utils

# Helper for parallel chunk processing


def _process_roi_chunk(
    Ly,
    Lx,
    sbin,
    filenames,
    cumframes,
    avgframe,
    avgmotion,
    U_mot,
    U_mov,
    motSVD,
    movSVD,
    t_start,
    t_end,
    rois,
    fullSVD,
    ivid,
    motind,
    MainWindow,
    GUIobject,
):
    """Processes a single, independent chunk of frames for parallel execution."""
    # Open containers locally in each worker
    _, _, _, containers = utils.get_frame_details(filenames)
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    nroi = len(motind)

    frames_to_load = np.arange(t_start, t_end)
    chunk_len_frames = len(frames_to_load)

    # If less than 2 frames, can't compute motion, return empty arrays with correct shapes
    if chunk_len_frames < 2:
        V_mot_chunk = (
            [np.zeros((0, U_mot[0].shape[1]), np.float32)]
            if motSVD and len(U_mot) > 0
            else []
        )
        V_mov_chunk = (
            [np.zeros((0, U_mov[0].shape[1]), np.float32)]
            if movSVD and len(U_mov) > 0
            else []
        )
        M_chunk = [np.zeros(0, np.float32)]
        utils.close_videos(containers)
        return V_mot_chunk, V_mov_chunk, M_chunk, None, None, None

    img = imall_init(chunk_len_frames, Ly, Lx)
    utils.get_frames(img, containers, frames_to_load, cumframes)

    output_len = chunk_len_frames - 1

    # Initialize outputs for this chunk
    V_mot_chunk = (
        [np.zeros((output_len, U_mot[0].shape[1]), np.float32)]
        if motSVD and len(U_mot) > 0
        else []
    )
    V_mov_chunk = (
        [np.zeros((output_len, U_mov[0].shape[1]), np.float32)]
        if movSVD and len(U_mov) > 0
        else []
    )
    M_chunk = [np.zeros((output_len,), np.float32)]

    # For fullSVD, initialize imall_mot and imall_mov
    if fullSVD and motSVD:
        imall_mot = np.zeros((output_len, (Lyb * Lxb).sum()), np.float32)
    else:
        imall_mot = None
    if fullSVD and movSVD:
        imall_mov = np.zeros((output_len, (Lyb * Lxb).sum()), np.float32)
    else:
        imall_mov = None

    # Process each video in this chunk
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
            if motSVD:
                imbin_mot = np.abs(np.diff(imbin, axis=0))
            if movSVD:
                imbin_mov = imbin[1:, :]

            if fullSVD:
                if motSVD and imall_mot is not None:
                    M_chunk[0] += imbin_mot.sum(axis=-1)
                    imall_mot[:, ir[ii]] = imbin_mot - avgmotion[ii].flatten()
                if movSVD and imall_mov is not None:
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
                ymin, ymax = (
                    rois[wroi[i]]["yrange_bin"][0],
                    rois[wroi[i]]["yrange_bin"][-1] + 1,
                )
                xmin, xmax = (
                    rois[wroi[i]]["xrange_bin"][0],
                    rois[wroi[i]]["xrange_bin"][-1] + 1,
                )
                if motSVD:
                    lilbin = imbin_mot[:, ymin:ymax, xmin:xmax]
                    M_chunk[0] += lilbin.sum(axis=(-2, -1))
                    lilbin -= avgmotion[ii][ymin:ymax, xmin:xmax]
                    lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                    vproj = lilbin @ U_mot[wmot[i] + 1]
                    V_mot_chunk[0] += vproj
                if movSVD:
                    lilbin = imbin_mov[:, ymin:ymax, xmin:xmax]
                    lilbin -= avgframe[ii][ymin:ymax, xmin:xmax]
                    lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                    vproj = lilbin @ U_mov[wmot[i] + 1]
                    V_mov_chunk[0] += vproj

    # For fullSVD, project onto U_mot[0] and U_mov[0]
    if fullSVD:
        if motSVD and imall_mot is not None:
            vproj = imall_mot @ U_mot[0]
            V_mot_chunk[0] = vproj
        if movSVD and imall_mov is not None:
            vproj = imall_mov @ U_mov[0]
            V_mov_chunk[0] = vproj

    # Close containers at the end
    utils.close_videos(containers)

    # For now, return None for pups_chunk, blinks_chunk, runs_chunk
    pups_chunk, blinks_chunk, runs_chunk = None, None, None

    return (
        V_mot_chunk,
        V_mov_chunk,
        M_chunk,
        pups_chunk,
        blinks_chunk,
        runs_chunk,
    )


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
    filenames,
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

    nroi = 0
    motind = []
    ivid = []
    if rois is not None:
        for i, r in enumerate(rois):
            ivid.append(r["ivid"])
            if r["rind"] == 1:
                motind.append(i)
                nroi += 1
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind)

    def _process_chunk_svd(n):
        _, _, _, containers = utils.get_frame_details(filenames)
        img = imall_init(nt0, Ly, Lx)
        t = tf[n]
        utils.get_frames(img, containers, np.arange(t, t + nt0), cumframes)

        mot_parts = [None] * (1 + nroi)
        mov_parts = [None] * (1 + nroi)

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

                        u_mot_idx = wmot[i] + 1

                        if motSVD:
                            lilbin = imbin_mot[:, ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            ncb = min(nc, lilbin.shape[-1])
                            usv = utils.svdecon(lilbin.T, k=ncb)
                            mot_parts[u_mot_idx] = usv[0] * usv[1]
                        if movSVD:
                            lilbin = imbin_mov[:, ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            ncb = min(nc, lilbin.shape[-1])
                            usv = utils.svdecon(lilbin.T, k=ncb)
                            mov_parts[u_mot_idx] = usv[0] * usv[1]

        if fullSVD:
            if motSVD:
                ncb = min(nc, imall_mot.shape[-1])
                usv = utils.svdecon(imall_mot.T, k=ncb)
                mot_parts[0] = usv[0] * usv[1]
            if movSVD:
                ncb = min(nc, imall_mov.shape[-1])
                usv = utils.svdecon(imall_mov.T, k=ncb)
                mov_parts[0] = usv[0] * usv[1]

        utils.close_videos(containers)
        return mot_parts, mov_parts

    w = StringIO()
    tic = time.time()

    # Process chunks in parallel with progress tracking
    print(f"Processing {nsegs} chunks in parallel...")
    with Parallel(n_jobs=-1) as parallel:
        results = []
        with tqdm(total=nsegs, desc="Computing SVD", file=sys.stdout) as pbar:
            for result in parallel(
                delayed(_process_chunk_svd)(n) for n in range(nsegs)
            ):
                results.append(result)
                pbar.update(1)

    U_mot = []
    U_mov = []
    if motSVD:
        U_mot = [
            np.concatenate(
                [res[0][i] for res in results if res[0][i] is not None], axis=1
            )
            for i in range(1 + nroi)
        ]
    if movSVD:
        U_mov = [
            np.concatenate(
                [res[1][i] for res in results if res[1][i] is not None], axis=1
            )
            for i in range(1 + nroi)
        ]

    S_mot = np.zeros(500, "float32")
    S_mov = np.zeros(500, "float32")
    # take SVD of concatenated spatial PCs
    if nsegs > 1:
        for nr in range(len(U_mot)):
            if U_mot[nr].shape[1] > 0:
                usv = utils.svdecon(
                    U_mot[nr],
                    k=min(ncomps, U_mot[nr].shape[0] - 1, U_mot[nr].shape[1] - 1),
                )
                U_mot[nr] = usv[0]
                if nr == 0:
                    S_mot = usv[1]

        for nr in range(len(U_mov)):
            if U_mov[nr].shape[1] > 0:
                usv = utils.svdecon(
                    U_mov[nr],
                    k=min(ncomps, U_mov[nr].shape[0] - 1, U_mov[nr].shape[1] - 1),
                )
                U_mov[nr] = usv[0]
                if nr == 0:
                    S_mov = usv[1]

    utils.update_mainwindow_message(MainWindow, GUIobject, "Finished computing svd")

    return U_mot, U_mov, S_mot, S_mov


def process_ROIs(
    filenames,  # changed from containers
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
):
    """
    Parallelized ROI processing with overlapping chunks to ensure correct output length.
    """
    nframes = cumframes[-1]
    nt0 = 500  # frames per chunk

    # Create overlapping chunk indices - ensure we cover all frames
    chunk_indices = []
    start = 0
    while start < nframes:
        end = min(start + nt0 + 1, nframes)  # +1 for overlap
        if end > start + 1:  # Need at least 2 frames for motion computation
            chunk_indices.append((start, end))
        if end >= nframes:
            break
        start += nt0

    ivid = []
    motind = []
    if rois is not None:
        for i, r in enumerate(rois):
            ivid.append(r["ivid"])
            if r["rind"] == 1:
                motind.append(i)
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind)

    # Process chunks in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_process_roi_chunk)(
            Ly,
            Lx,
            sbin,
            filenames,
            cumframes,
            avgframe,
            avgmotion,
            U_mot,
            U_mov,
            motSVD,
            movSVD,
            t_start,
            t_end,
            rois,
            fullSVD,
            ivid,
            motind,
            MainWindow,
            GUIobject,
        )
        for t_start, t_end in tqdm(chunk_indices, desc="Processing ROI chunks")
    )

    # Unzip results
    V_mot_list, V_mov_list, M_list, _, _, _ = zip(*results)

    # Concatenate outputs
    def concat_outputs(chunk_list):
        if not chunk_list or not any(item is not None for item in chunk_list):
            return []

        # Check if we have a list of lists of arrays
        if isinstance(chunk_list[0], list):
            num_outputs = len(chunk_list[0])
            final_outputs = []
            for i in range(num_outputs):
                # Collect all i-th arrays from each chunk
                arrays_to_concat = [
                    chunk[i]
                    for chunk in chunk_list
                    if chunk and len(chunk) > i and chunk[i] is not None
                ]
                if arrays_to_concat:
                    final_outputs.append(np.concatenate(arrays_to_concat, axis=0))
                else:
                    # Attempt to create an empty array with correct dimensions
                    template_chunk = next(
                        (
                            c
                            for c in chunk_list
                            if c and len(c) > i and c[i] is not None
                        ),
                        None,
                    )
                    if template_chunk is not None:
                        final_outputs.append(
                            np.zeros(
                                (0,) + template_chunk[i].shape[1:],
                                dtype=template_chunk[i].dtype,
                            )
                        )
                    else:
                        final_outputs.append(np.array([]))
            return final_outputs
        # Fallback for simple list of arrays
        elif isinstance(chunk_list[0], np.ndarray):
            return np.concatenate(chunk_list, axis=0)
        return []

    V_mot = concat_outputs(V_mot_list)
    V_mov = concat_outputs(V_mov_list)
    M = concat_outputs(
        list(M_list)
    )  # M_list is a tuple of lists, convert to list of lists

    # Repeat first value and ensure length is nframes (to match original behavior)
    if motSVD and V_mot and len(V_mot) > 0 and V_mot[0].shape[0] > 0:
        # Prepend the first value to get length nframes
        first_val = V_mot[0][0:1]  # Keep dimensions
        V_mot[0] = np.concatenate([first_val, V_mot[0]], axis=0)
        # Trim to exact length nframes
        if V_mot[0].shape[0] > nframes:
            V_mot[0] = V_mot[0][:nframes]

    if movSVD and V_mov and len(V_mov) > 0 and V_mov[0].shape[0] > 0:
        # Prepend the first value to get length nframes
        first_val = V_mov[0][0:1]  # Keep dimensions
        V_mov[0] = np.concatenate([first_val, V_mov[0]], axis=0)
        # Trim to exact length nframes
        if V_mov[0].shape[0] > nframes:
            V_mov[0] = V_mov[0][:nframes]

    if M and len(M) > 0 and M[0].shape[0] > 0:
        # Prepend the first value to get length nframes
        first_val = M[0][0:1]  # Keep dimensions
        M[0] = np.concatenate([first_val, M[0]], axis=0)
        # Trim to exact length nframes
        if M[0].shape[0] > nframes:
            M[0] = M[0][:nframes]

    utils.update_mainwindow_message(
        MainWindow, GUIobject, "Finished computing ROIs and/or motSVD/movSVD "
    )
    # Return empty lists for non-implemented features
    pups, blinks, runs = [], [], []
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
        Spatial binning factor. If sbin > 1, the movie will be spatially binning by a factor of sbin.
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
            savepath = (
                proc["savepath"] if savepath is None else savepath
            )  # proc["savepath"] if savepath is not None else savepath
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
            filenames,
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
        print(
            "Computed subsampled SVD at %0.2fs" % (time.time() - tic), file=sys.stdout
        )

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
    V_mot, V_mov, M, pups, blinks, runs = process_ROIs(
        filenames,  # pass filenames, not containers
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
    )
    print(
        "Computed ROIS and/or motSVD/movSVD at %0.2fs" % (time.time() - tic),
        file=sys.stdout,
    )

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
