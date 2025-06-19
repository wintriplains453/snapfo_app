#ifndef FACE_ALIGNMENT_H #define FACE_ALIGNMENT_H

#ifdef __cplusplus extern "C" { #endif

char* align_face_ffi(const double* landmarks, int landmark_count, const char* filepath); void free_string(char* str);

#ifdef __cplusplus } #endif

#endif