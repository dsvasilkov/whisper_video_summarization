import { FFmpeg, FFFSType } from '@ffmpeg/ffmpeg'
import { toBlobURL } from '@ffmpeg/util'

const ffmpeg = new FFmpeg()
let ffmpegLoaded: Promise<void> | null = null

async function ensureFfmpegLoaded(): Promise<void> {
  if (ffmpegLoaded) return ffmpegLoaded

  ffmpegLoaded = (async () => {
    const base = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/esm'
    const coreURL = await toBlobURL(`${base}/ffmpeg-core.js`, 'text/javascript')
    const wasmURL = await toBlobURL(`${base}/ffmpeg-core.wasm`, 'application/wasm')
    await ffmpeg.load({ coreURL, wasmURL })
  })()

  return ffmpegLoaded
}

/** Virtual FS name + demuxer hints so FFmpeg recognizes Matroska and other containers. */
function virtualInputSpec(file: File): { path: string; demuxPrefix: string[] } {
  const stamp = Date.now()
  const base = `input_${stamp}`
  const lower = file.name.toLowerCase()
  const ext = lower.match(/(\.[a-z0-9]+)$/)?.[1] ?? ''
  const mime = file.type.toLowerCase()

  if (ext === '.mkv' || ext === '.mka' || mime === 'video/x-matroska') {
    return { path: `${base}.mkv`, demuxPrefix: ['-f', 'matroska'] }
  }
  if (ext === '.webm' || mime === 'video/webm') {
    return { path: `${base}.webm`, demuxPrefix: ['-f', 'webm'] }
  }
  if (/^\.(mp4|m4v|mov|avi|wav|mp3|m4a|aac|ogg|opus|flac)$/.test(ext)) {
    return { path: base + ext, demuxPrefix: [] }
  }

  const mimeExt: Record<string, string> = {
    'video/mp4': '.mp4',
    'video/quicktime': '.mov',
    'video/x-msvideo': '.avi',
    'audio/mpeg': '.mp3',
    'audio/wav': '.wav',
    'audio/x-wav': '.wav',
    'audio/mp4': '.m4a',
    'audio/aac': '.aac',
    'audio/ogg': '.ogg',
    'audio/opus': '.opus',
    'audio/flac': '.flac',
  }
  const mapped = mimeExt[mime]
  if (mapped) return { path: base + mapped, demuxPrefix: [] }

  return { path: ext ? base + ext : `${base}.bin`, demuxPrefix: [] }
}

export async function convertMediaToWav(file: File): Promise<File> {
  await ensureFfmpegLoaded()

  const { path: inputName, demuxPrefix } = virtualInputSpec(file)
  const outputName = `output_${Date.now()}.wav`
  // Fresh path each run: mkdir must exist before mount or Emscripten throws ErrnoError: FS error.
  const inputMount = `/wfs_in_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`
  const inputPath = `${inputMount}/${inputName}`

  try {
    await ffmpeg.createDir(inputMount)
    // WORKERFS: FFmpeg reads from the file handle — no giant JS ArrayBuffer for the source video.
    await ffmpeg.mount(FFFSType.WORKERFS, { blobs: [{ name: inputName, data: file }] }, inputMount)

    await ffmpeg.exec([
      ...demuxPrefix,
      '-i',
      inputPath,
      // Use a deterministic audio stream and preserve timeline continuity.
      '-map',
      '0:a:0',
      '-vn',
      '-af',
      'aresample=async=1:first_pts=0',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-acodec',
      'pcm_s16le',
      outputName,
    ])

    const data = await ffmpeg.readFile(outputName)
    await ffmpeg.deleteFile(outputName)

    const stem = file.name.replace(/\.[^/.]+$/, '') || 'audio'
    const wavBytes = new Uint8Array(data as Uint8Array)
    return new File([wavBytes], `${stem}.wav`, { type: 'audio/wav' })
  } finally {
    try {
      await ffmpeg.unmount(inputMount)
    } catch {
      /* mount failed or already unmounted */
    }
    try {
      await ffmpeg.deleteDir(inputMount)
    } catch {
      /* not empty or missing */
    }
  }
}
