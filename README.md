# Multi-feature based network for multivariate time series classification

**The repository is for a paper on this topic under review.**

Multivariate time series classification is widely available in several areas of real life and has attracted the attention of many researchers. In recent years, a large number of multivariate time series classification algorithms have been proposed. However, existing multivariate time series classification methods focus only on local or global features and usually ignore the spatial dependency features among multiple variables. For this, a Multi-Feature based Network (MF-Net) is proposed. Firstly, MF-Net uses the global-local block to acquire local features through the attention-based mechanism. Next, the sparse self-attention mechanism captures global features. Finally, MF-Net integrates the local features and global features to capture the spatial dependency features through the spatial-local block. Therefore, we are able to mine the spatial dependency features of multivariate time series while incorporating both local and global features. We perform experiments on UEA datasets, and the experimental results show that our method obtains a competitive performance with state-of-the-art methods.

## Requirements
* Python 
* PyTorch 

## Datasets
Get MTS datasets in http://timeseriesclassification.com/dataset.php.

**Train/Test**:

```bash
python run_UEA.py
```

## Acknowledgements
This work was supported by the Innovation Methods Work Special Project under Grant 2020IM020100, and the Natural Science Foundation of Shandong Province under Grant ZR2020QF112.

We would like to thank Eamonn Keogh and his team, Tony Bagnall and his team for the UEA/UCR time series classification repository.
