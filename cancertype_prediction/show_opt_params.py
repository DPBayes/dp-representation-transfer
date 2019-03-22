import os.path
import numpy as np

cancer_type_pairs = [
  ["lung squamous cell carcinoma", "head & neck squamous cell carcinoma"],
  ["bladder urothelial carcinoma", "cervical & endocervical cancer"],
  ["colon adenocarcinoma", "rectum adenocarcinoma"],
  ["stomach adenocarcinoma", "esophageal carcinoma"],
#  ["lung squamous cell carcinoma", "head & neck squamous cell carcinoma"],
#  ["kidney clear cell carcinoma", "kidney papillary cell carcinoma"],
#  ["colon adenocarcinoma", "rectum adenocarcinoma"],
]

priv_val_pairs = [
  (cancer_type_pairs[0], cancer_type_pairs[1]),
  (cancer_type_pairs[0], cancer_type_pairs[2]),
  (cancer_type_pairs[1], cancer_type_pairs[0]),
  (cancer_type_pairs[1], cancer_type_pairs[2]),
  (cancer_type_pairs[2], cancer_type_pairs[0]),
  (cancer_type_pairs[2], cancer_type_pairs[1]),
  (cancer_type_pairs[0], cancer_type_pairs[3]),
  (cancer_type_pairs[3], cancer_type_pairs[0]),
  (cancer_type_pairs[1], cancer_type_pairs[3]),
  (cancer_type_pairs[3], cancer_type_pairs[1]),
  (cancer_type_pairs[2], cancer_type_pairs[3]),
  (cancer_type_pairs[3], cancer_type_pairs[2]),
  (["lung squamous cell carcinoma", "head & neck squamous cell carcinoma"], ["kidney clear cell carcinoma", "kidney papillary cell carcinoma"]),
]

algs = [
  ('rand_proj',{}),
  ('PCA',{}),
  ('VAE',{}),
 ]

test_id = ""

def np_loadtxt_or(filename, fallback):
  if os.path.isfile(filename) and os.path.getsize(filename) > 0:
    return np.loadtxt(filename)
  else:
    print("    Warning: File not found or empty: %s" % (filename))
    return fallback

for pv, (priv, val) in enumerate(priv_val_pairs):
  print("priv = %s" % priv)
  print("val = %s" % val)
  data_name = (('-'.join(['priv',] + priv +
                ['val',] + val))
                .replace(' ', '_').replace('&', '_'))
  for a, (repr_alg, style_args) in enumerate(algs):
    test_name = "%s%s-%s" % (test_id, data_name, repr_alg)
    params = np.load("param_opt/opt_params-%s.npy" % (test_name))
    results = np.load("param_opt/opt_results-%s.npy" % (test_name))
    i = np.argmax(results)
    print("%s: %d tested, best: %s -> %s" % (repr_alg, len(results), params[i], results[i,0]))
    #print(np.hstack((params, results)))
  print()