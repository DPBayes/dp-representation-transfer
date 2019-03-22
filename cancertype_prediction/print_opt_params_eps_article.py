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

priv = cancer_type_pairs[0]
epsilons = [0.5, 0.7, 1.0, 1.5, 2.0]

algs = [
  ('rand_proj', "RP", {}),
  ('PCA', "PCA", {}),
  ('VAE', "VAE", {}),
  #('VAE_hyper',{}),
 ]

test_id = ""

def np_loadtxt_or(filename, fallback):
  if os.path.isfile(filename) and os.path.getsize(filename) > 0:
    return np.loadtxt(filename)
  else:
    print("    Warning: File not found or empty: %s" % (filename))
    return fallback

print("\\begin{tabular}{ c | ", end='')
for a, (repr_alg, alg_name, style_args) in enumerate(algs):
  print("c ", end='')
print("c ", end='')
print("c ", end='')
print("c ", end='')
print("}")

print("  \\cline{2-7}")
print("  ", end='')
for a, (repr_alg, alg_name, style_args) in enumerate(algs):
  #print(" & %s" % (alg_name), end='')
  print(" & \\multicolumn{%d}{c|}{%s}" % (4 if repr_alg == 'VAE' else 1, alg_name), end='')
print(" \\\\")
#print("  \\hline")
print("  \\cline{2-7}")

#print("  ", end='')
print("  $\epsilon$", end='')
print(" & \\multicolumn{3}{c|}{repr-dim}", end='')	
print(" & \\multicolumn{1}{c|}{log-lr}", end='')
print(" & \\multicolumn{1}{c|}{layers}", end='')
print(" & \\multicolumn{1}{c|}{layer-dim}", end='')
print(" \\\\")
print("  \\hline")

for eps in epsilons:
  #print("priv = %s" % priv)
  print("  %g" % eps, end='')
  data_name = (('-'.join(['priv',] + priv)).replace(' ', '_').replace('&', '_'))
  for a, (repr_alg, alg_name, style_args) in enumerate(algs):
    test_name = "%s%s-%s-e%g" % (test_id, data_name, repr_alg, eps)
    params_filename = "param_opt/opt_params-%s.npy" % (test_name)
    results_filename = "param_opt/opt_results-%s.npy" % (test_name)
    if (os.path.isfile(params_filename) and os.path.getsize(params_filename) > 0 and
        os.path.isfile(results_filename) and os.path.getsize(results_filename) > 0):
      params = np.load(params_filename)
      results = np.load(results_filename)
      i = np.argmax(results)
      print(" & %d" % params[i,0], end='')
      #print("%s: %d tested, best: %s -> %s" %
      #      (repr_alg, len(results), params[i], results[i,0]))
      #print(np.hstack((params, results)))
    else:
      #print("%s: Error: param and/or result files not found" % (repr_alg))
      pass
  assert repr_alg == 'VAE'
  print(" & %.1f" % params[i,1], end='')
  print(" & %d" % params[i,2], end='')
  #print(" & %.1f" % params[i,3], end='')
  print(" & %d" % (int(10 ** params[i,3])*params[i,0]), end='')
  print(" \\\\")
  #print()

print("\\end{tabular}")
