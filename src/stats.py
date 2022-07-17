from __future__ import annotations
from BratAnnotationReader import BratAnnotationReader

bratReader = BratAnnotationReader('/home/sonic/uw/600/brat-1.3_Crunchy_Frog/data/cos_data/snippet1_100.ann')
annotations = bratReader.getAnnotations()
print(annotations)