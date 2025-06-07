# our module with demo
MOD_NAME := "mvp"

# Convert realsense video to output pose estimation tensor
.PHONY: trainer_LSTM
trainer_LSTM: bag2mp4 yolo17 get3d get_features LSTM

.PHONY: trainer_MLP
trainer_MLP: bag2mp4 yolo17 get3d get_features MLP

.PHONY: trainer_Catboost
trainer_Catboost: bag2mp4 yolo17 get3d get_features Catboost

.PHONY: trainer_LGBM
trainer_LGBM: bag2mp4 yolo17 get3d get_features LGBM

.PHONY: trainer_Xgboost
trainer_Xgboost: bag2mp4 yolo17 get3d get_features Xgboost

.PHONY: eval_LSTM
eval_LSTM: LSTM_eval

.PHONY: eval_MLP
eval_MLP: MLP_eval

.PHONY: eval_Catboost
eval_Catboost: Catboost_eval

.PHONY: eval_LGBM
eval_LGBM: LGBM_eval

.PHONY: eval_Xgboost
eval_Xgboost: Xgboost_eval
# -----------------------

.PHONY: bag2mp4
bag2mp4:
	python3 -m "${MOD_NAME}".bag2mp4

.PHONY: yolo17
yolo17:
	python3 -m "${MOD_NAME}".yolo17

.PHONY: yolo35
yolo35:
	python3 -m "${MOD_NAME}".yolo35

.PHONY: get3d
get3d:
	python3 -m "${MOD_NAME}".get3d

.PHONY: get_features
get_features:
	python3 -m "${MOD_NAME}".feature_bake

.PHONY: LSTM
LSTM:
	python3 -m "${MOD_NAME}".models.LSTM

.PHONY: Xgboost
Xgboost:
	python3 -m "${MOD_NAME}".models.Xgboost

.PHONY: MLP
MLP:
	python3 -m "${MOD_NAME}".models.MLP

.PHONY: Catboost
Catboost:
	python3 -m "${MOD_NAME}".models.Catboost

.PHONY: LGBM
LGBM:
	python3 -m "${MOD_NAME}".models.LGBM


.PHONY: LSTM_evals
LSTM_eval:
	python3 -m "${MOD_NAME}".models.LSTM.evaluation

.PHONY: plot_builder
plot_builder:
	python3 -m "${MOD_NAME}".compare_features


.PHONY: LGBM_eval
LGBM_eval:
	python3 -m "${MOD_NAME}".models.LGBM.evaluation

.PHONY: Xgboost_eval
Xgboost_eval:
	python3 -m "${MOD_NAME}".models.Xgboost.evaluation

.PHONY: MLP_eval
MLP_eval:
	python3 -m "${MOD_NAME}".models.MLP.evaluation

.PHONY: Catboost_eval
Catboost_eval:
	python3 -m "${MOD_NAME}".models.Catboost.evaluation
