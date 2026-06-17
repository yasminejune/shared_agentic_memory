# Run compare_clustering_methods.py

make install                                # picks up umap-learn + hdbscan
venv/bin/python -m scripts.amin_et_al.compare_clustering_methods \
    --methods bertopic bertopic_pure  