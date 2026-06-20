#include <stddef.h>

#include "doomtype.h"
#include "i_sound.h"

static boolean Null_SoundInit(boolean use_sfx_prefix)
{
    (void)use_sfx_prefix;
    return true;
}

static void Null_SoundShutdown(void) {}

static int Null_GetSfxLumpNum(sfxinfo_t *sfx)
{
    (void)sfx;
    return 0;
}

static void Null_SoundUpdate(void) {}

static void Null_UpdateSoundParams(int ch, int vol, int sep)
{
    (void)ch;
    (void)vol;
    (void)sep;
}

static int Null_StartSound(sfxinfo_t *sfx, int ch, int vol, int sep)
{
    (void)sfx;
    (void)ch;
    (void)vol;
    (void)sep;
    return -1;
}

static void Null_StopSound(int ch) { (void)ch; }

static boolean Null_SoundIsPlaying(int ch)
{
    (void)ch;
    return false;
}

static void Null_CacheSounds(sfxinfo_t *sounds, int num)
{
    (void)sounds;
    (void)num;
}

static snddevice_t sound_devices[] = { SNDDEVICE_SB };

sound_module_t DG_sound_module =
{
    sound_devices,
    1,
    Null_SoundInit,
    Null_SoundShutdown,
    Null_GetSfxLumpNum,
    Null_SoundUpdate,
    Null_UpdateSoundParams,
    Null_StartSound,
    Null_StopSound,
    Null_SoundIsPlaying,
    Null_CacheSounds,
};

static boolean Null_MusicInit(void) { return true; }
static void Null_MusicShutdown(void) {}
static void Null_SetMusicVolume(int v) { (void)v; }
static void Null_PauseMusic(void) {}
static void Null_ResumeMusic(void) {}
static void *Null_RegisterSong(void *data, int len)
{
    (void)data;
    (void)len;
    return NULL;
}
static void Null_UnRegisterSong(void *h) { (void)h; }
static void Null_PlaySong(void *h, boolean looping)
{
    (void)h;
    (void)looping;
}
static void Null_StopSong(void) {}
static boolean Null_MusicIsPlaying(void) { return false; }
static void Null_PollMusic(void) {}

music_module_t DG_music_module =
{
    sound_devices,
    1,
    Null_MusicInit,
    Null_MusicShutdown,
    Null_SetMusicVolume,
    Null_PauseMusic,
    Null_ResumeMusic,
    Null_RegisterSong,
    Null_UnRegisterSong,
    Null_PlaySong,
    Null_StopSong,
    Null_MusicIsPlaying,
    Null_PollMusic,
};
