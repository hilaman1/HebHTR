#include "Beam.hpp"
#include "LanguageModel.hpp"
#include "PrefixTree.hpp"
#include "WordBeamSearch.hpp"
#include <Python.h>

PyMODINIT_FUNC PyInit_word_beam_search_module(void);

/*---------- Module initialization methods ----------*/

static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT,
        "word_beam_search_module",
        NULL,
        -1,
};

PyMODINIT_FUNC PyInit_word_beam_search_module(void){
    PyObject *m;
    m=PyModule_Create(&module);
    if (!m) {
        return NULL;
    }
    return m;
    }
