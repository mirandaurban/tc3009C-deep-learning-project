import unittest
import numpy as np
import pandas as pd
from src.features import get_features, validate_predictors
from src.pipeline import split_by_runner
from src.preprocessing import fit_transform_pair
from src.optimization import sample_configs
from src.evaluation import metrics
from src.training import fit_mlp, predict

class ComponentsTest(unittest.TestCase):
    def test_feature_integrity(self):
        self.assertNotIn("id",get_features("B")); self.assertEqual(len(get_features("B")),12)
        with self.assertRaises(ValueError): validate_predictors(["id"])

    def test_group_split_is_disjoint(self):
        frame=pd.DataFrame({"id":np.repeat(np.arange(40),2),"x":np.arange(80)})
        parts=split_by_runner(frame)
        ids=[set(p.id) for p in parts]
        self.assertTrue(all(not ids[i]&ids[j] for i in range(3) for j in range(i+1,3)))

    def test_scaler_and_ridge_compatible(self):
        train=pd.DataFrame({"age":[1.,2.,3.],"bmi":[20.,21.,22.]})
        other=pd.DataFrame({"age":[4.],"bmi":[23.]})
        x,y,_=fit_transform_pair(train,other,["age","bmi"])
        np.testing.assert_allclose(x.mean(axis=0),0,atol=1e-12)
        self.assertEqual(y.shape,(1,2))

    def test_search_reproducible_and_metrics(self):
        self.assertEqual(sample_configs(5,9),sample_configs(5,9))
        self.assertAlmostEqual(metrics([1,2],[1,2])["RMSE"],0)

    def test_mlp_isolated_synthetic_fit(self):
        x=np.arange(40,dtype=float).reshape(-1,1)/40; y=x[:,0]*2
        model,hist,epoch=fit_mlp(x[:30],y[:30],x[30:],y[30:],seed=3,hidden=(4,),epochs=2,batch_size=10)
        self.assertEqual(len(hist["val"]),2); self.assertLess(epoch,2)
        self.assertEqual(predict(model,x[30:]).shape,(10,))

if __name__=="__main__": unittest.main()
