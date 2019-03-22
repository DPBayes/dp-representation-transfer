import os.path
import numpy as np

cancer_type_pairs = [
  ["lung squamous cell carcinoma", "head & neck squamous cell carcinoma"],
  ["bladder urothelial carcinoma", "cervical & endocervical cancer"],
  ["colon adenocarcinoma", "rectum adenocarcinoma"],
  ["stomach adenocarcinoma", "esophageal carcinoma"],
  ["kidney clear cell carcinoma", "kidney papillary cell carcinoma"],
  ["glioblastoma multiforme", "sarcoma"],
  ["adrenocortical cancer", "uveal melanoma"],
  ["testicular germ cell tumor", "uterine carcinosarcoma"],
  ["lung adenocarcinoma", "pancreatic adenocarcinoma"],
  ["ovarian serous cystadenocarcinoma", "uterine corpus endometrioid carcinoma"],
  ["brain lower grade glioma", "pheochromocytoma & paraganglioma"],
  ["skin cutaneous melanoma", "mesothelioma"],
  ["liver hepatocellular carcinoma", "kidney chromophobe"],
  ["breast invasive carcinoma", "prostate adenocarcinoma"],
  ["acute myeloid leukemia", "diffuse large B-cell lymphoma"],
  ["thyroid carcinoma", "cholangiocarcinoma"],
]

priv_pairs = [
  cancer_type_pairs[0],
  cancer_type_pairs[1],
  cancer_type_pairs[2],
  cancer_type_pairs[3],
  cancer_type_pairs[4],
  cancer_type_pairs[5],
  cancer_type_pairs[9],
  cancer_type_pairs[13],
]

algs = [
  ('rand_proj',{}),
  ('PCA',{}),
  ('VAE',{}),
  ('VAE_hyper',{}),
 ]

test_id = ""

def np_loadtxt_or(filename, fallback):
  if os.path.isfile(filename) and os.path.getsize(filename) > 0:
    return np.loadtxt(filename)
  else:
    print("    Warning: File not found or empty: %s" % (filename))
    return fallback

for pv, priv in enumerate(priv_pairs):
  print("priv = %s" % priv)
  data_name = (('-'.join(['priv',] + priv)).replace(' ', '_').replace('&', '_'))
  for a, (repr_alg, style_args) in enumerate(algs):
    test_name = "%s%s-%s" % (test_id, data_name, repr_alg)
    params_filename = "param_opt/opt_params-%s.npy" % (test_name)
    results_filename = "param_opt/opt_results-%s.npy" % (test_name)
    if (os.path.isfile(params_filename) and os.path.getsize(params_filename) > 0 and
        os.path.isfile(results_filename) and os.path.getsize(results_filename) > 0):
      params = np.load(params_filename)
      results = np.load(results_filename)
      i = np.argmax(results)
      print("%s: %d tested, best: %s -> %s" %
            (repr_alg, len(results), params[i], results[i,0]))
      #print(np.hstack((params, results)))
    else:
      print("%s: Error: param and/or result files not found" % (repr_alg))
  print()
