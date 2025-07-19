# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the boundary attack `BoundaryAttack`. This is a black-box attack which only requires class
predictions.

| Paper link: https://arxiv.org/abs/1712.04248
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import numpy as np
from tqdm.auto import tqdm, trange

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format, get_labels_np_array

from models.art_statefuldefense import ArtStatefulDefense
from IPython import embed
from attacks.Attack import Attack

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class Boundary(Attack):
    def __init__(self, model, model_config, attack_config):
        self.model_config = model_config
        self.attack_config = attack_config
        self.model_art = ArtStatefulDefense(model=model, device_type='gpu',
                                            input_shape=model_config['state']['input_shape'], loss=None,
                                            nb_classes=attack_config['nb_classes'])
        self.art_attack = BoundaryAttack(estimator=self.model_art, model_config=model_config, batch_size=1,
                                               targeted=False,
                                               min_epsilon=attack_config["eps"], attack_config=attack_config)
        self._model = self.model_art._model._model



    # 带有目标标签的攻击
    def attack_targeted(self, x, y, x_adv):
        # 将输入图像 x 和标签 y 从 GPU 转移到 CPU 并转换为 NumPy 数组
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()[0]

        # 确保输入的图像 x 是单张图像 (batch size = 1)
        assert x_np.shape[0] == 1

        # 创建一个 one-hot 编码的标签向量，用于表示目标标签
        one_hot_labels = torch.zeros((1, self.attack_config['nb_classes']))
        one_hot_labels[0, y_np] = 1  # 设置目标标签的位置为 1

        # 使用目标攻击生成对抗样本
        x_adv_np = self.art_attack.generate(x=x_np, y=one_hot_labels, x_adv_init=x_adv)

        # 如果生成的对抗样本是一个错误信息字符串，则结束攻击
        if isinstance(x_adv_np, str):
            self.end(x_adv_np)

        # 计算对抗样本和原始图像之间的 L2 范数（归一化）
        if np.linalg.norm(x_adv_np - x_np) / (x_np.shape[-1] * x_np.shape[-2] * x_np.shape[-3]) ** 0.5 < \
                self.attack_config["eps"]:

            return torch.tensor(x_adv_np).cuda()

        # 否则，返回原始图像
        return torch.tensor(x_np).cuda()

    def attack_untargeted(self, x, y):
        # 将输入图像 x 和标签 y 从 GPU 转移到 CPU 并转换为 NumPy 数组
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()[0]

        # 确保输入的图像 x 是单张图像 (batch size = 1)
        assert x_np.shape[0] == 1

        # 创建一个 one-hot 编码的标签向量，用于表示真实标签
        one_hot_labels = torch.zeros((1, self.attack_config['nb_classes']))
        one_hot_labels[0, y_np] = 1  # 设置真实标签的位置为 1

        # 使用无目标攻击生成对抗样本
        x_adv_np = self.art_attack.generate(x=x_np, y=one_hot_labels, x_adv_init=None)

        # 如果生成的对抗样本是一个错误信息字符串，则结束攻击
        if isinstance(x_adv_np, str):
            self.end(x_adv_np)

        # 计算对抗样本和原始图像之间的 L2 范数（归一化）
        if np.linalg.norm(x_adv_np - x_np) / (x_np.shape[-1] * x_np.shape[-2] * x_np.shape[-3]) ** 0.5 < \
                self.attack_config["eps"]:
            # 如果范数小于设定的 epsilon，表示攻击成功，返回对抗样本
            return torch.tensor(x_adv_np).cuda()

        # 否则，返回原始图像
        return torch.tensor(x_np).cuda()


class BoundaryAttack(EvasionAttack):
    """
    Implementation of the boundary attack from Brendel et al. (2018). This is a powerful black-box attack that
    only requires final class prediction.

    | Paper link: https://arxiv.org/abs/1712.04248
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "delta",
        "epsilon",
        "step_adapt",
        "max_iter",
        "num_trial",
        "sample_size",
        "init_size",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
            self,
            estimator: "CLASSIFIER_TYPE",
            batch_size: int = 64,
            targeted: bool = True,
            delta: float = 0.01,
            epsilon: float = 0.01,
            step_adapt: float = 0.667,
            max_iter: int = 10000,
            num_trial: int = 10,
            sample_size: int = 20,
            init_size: int = 100,
            min_epsilon: float = 0.0,
            model_config: dict = None,
            attack_config: dict = None,
            verbose: bool = True
    ) -> None:
        """
        Create a boundary attack instance.

        :param estimator: A trained classifier.
        :param batch_size: The size of the batch used by the estimator during inference.
        :param targeted: Should the attack target one specific class.
        :param delta: Initial step size for the orthogonal step.
        :param epsilon: Initial step size for the step towards the target.
        :param step_adapt: Factor by which the step sizes are multiplied or divided, must be in the range (0, 1).
        :param max_iter: Maximum number of iterations.
        :param num_trial: Maximum number of trials per iteration.
        :param sample_size: Number of samples per trial.
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :param min_epsilon: Stop attack if perturbation is smaller than `min_epsilon`.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)

        self._targeted = targeted
        self.delta = delta
        self.epsilon = epsilon
        self.step_adapt = step_adapt
        self.max_iter = max_iter
        self.num_trial = num_trial
        self.sample_size = sample_size
        self.init_size = init_size
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_config = model_config
        self.attack_config = attack_config
        self._check_params()

        self.curr_adv: Optional[np.ndarray] = None
        from utils.logger import get_attack_logger
        self.boundary_logger = get_attack_logger("boundary")


    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels.
        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.
        :type x_adv_init: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        assert x.shape[0] == 1

        if y is None:
            raise NotImplementedError
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)

        if y is not None and self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(  # pragma: no cover
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Get clip_min and clip_max from the classifier or infer them from data
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # Prediction from the original images
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size)[0], axis=1)
        #########self.estimator._model._model.reset()

        init_preds = [None] * len(x)
        x_adv_init = [None] * len(x)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)

        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="Boundary attack", disable=not self.verbose)):
            if self.targeted:
                out = self._perturb(
                    x=val,
                    y=y[ind],
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
                if isinstance(out, str):
                    return out
                x_adv[ind] = out
            else:
                out = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
                if isinstance(out, str):
                    return out
                x_adv[ind] = out

        y = to_categorical(y, self.estimator.nb_classes)

        return x_adv

    def _perturb(
            self,
            x: np.ndarray,
            y: int,
            y_p: int,
            init_pred: int,
            adv_init: np.ndarray,
            clip_min: float,
            clip_max: float,
    ) -> np.ndarray:
        """
        Internal attack function for one example.

        :param x: An array with one original input to be attacked.
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :param y_p: The predicted label of x.
        :param init_pred: The predicted label of the initial image.
        :param adv_init: Initial array to act as an initial adversarial example.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: An adversarial example.
        """
        # First, create an initial adversarial sample

        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, clip_min, clip_max)
        if isinstance(initial_sample, str):
            return initial_sample

        # If an initial adversarial example is not found, then return the original image
        if initial_sample is None:
            return x

        # If an initial adversarial example found, then go with boundary attack
        x_adv = self._attack(
            initial_sample[0],
            x,
            y_p,
            initial_sample[1],
            self.delta,
            self.epsilon,
            clip_min,
            clip_max,
        )

        return x_adv

    def _attack(
            self,
            initial_sample: np.ndarray,  # 初始对抗样本（需满足攻击成功）
            original_sample: np.ndarray,  # 原始输入样本
            y_p: int,  # 原始输入样本的预测标签
            target: int,  # 攻击目标标签（若为非定向攻击，可忽略）
            initial_delta: float,  # 正交扰动初始步长
            initial_epsilon: float,  # 向目标方向扰动初始步长
            clip_min: float,  # 输入值的最小值（如像素值最小值）
            clip_max: float,  # 输入值的最大值（如像素值最大值）
    ) -> np.ndarray:
        """
        Main function for the boundary attack.

        :param initial_sample: An initial adversarial example.
        :param original_sample: The original input.
        :param y_p: The predicted label of the original input.
        :param target: The target label.
        :param initial_delta: Initial step size for the orthogonal step.
        :param initial_epsilon: Initial step size for the step towards the target.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: an adversarial example.
        """
        # Get initialization for some variables
        x_adv = initial_sample
        self.curr_delta = initial_delta
        self.curr_epsilon = initial_epsilon

        self.curr_adv = x_adv

        best_l2 = 10000000      # 当前找到的最优L2距离
        best_l2_set = 0         # 最优L2所在的迭代次数
        # Main loop to wander around the boundary
        # 主循环：围绕边界随机游走，寻找更优对抗样本
        pbar = trange(self.max_iter, leave=True)
        for this_iter in pbar:
            # 如果在50次内没有更新最优L2，则提前终止
            if this_iter - best_l2_set >= 50:
                break
            # Trust region method to adjust delta
            # Trust region：根据效果自适应调整 delta（正交扰动大小）
            for _ in range(self.num_trial):
                potential_advs = []
                for _ in range(self.sample_size):
                    # 添加正交扰动
                    potential_adv = x_adv + self._orthogonal_perturb(self.curr_delta, x_adv, original_sample)
                    potential_adv = np.clip(potential_adv, clip_min, clip_max)
                    potential_advs.append(potential_adv)
                # Don't let attacks go further than they need to, because it may make it un-adversarial vs cache.

                # 预测扰动后的样本
                out_preds, cache_preds = self.estimator.predict(np.array(potential_advs), batch_size=self.batch_size)
                preds = np.argmax(
                    out_preds,
                    axis=1,
                )
                # 若拒识样本不允许调整 delta，则直接返回
                if not self.attack_config["adaptive"]["delta_adjust_rejected_predictions"] and np.sum(np.where(cache_preds[:, 0], 1, 0)) > 0:
                    return "rejection in delta step"

                # 针对性攻击或非定向攻击分类处理
                if self.attack_config["targeted"]:
                    preds = np.where(cache_preds[:, 0], target, preds)
                else:
                    preds = np.where(cache_preds[:, 0], -1, preds)

                if self.attack_config["targeted"]:
                    is_adv = preds == target
                else:
                    is_adv = preds != y_p

                # 计算成功率比值
                if self.attack_config["adaptive"]["delta_extraction"]:
                    delta_ratio = np.mean(is_adv)
                else:
                    delta_ratio = np.mean(is_adv[~np.array(cache_preds[:, 0])])

                # 根据成功率动态调整 delta
                if delta_ratio < 0.2:
                    self.curr_delta *= self.step_adapt
                elif delta_ratio > 0.5:
                    self.curr_delta /= self.step_adapt

                # 找到满足条件的对抗样本
                delta_good = is_adv * (preds >= 0) * (~np.array(cache_preds[:, 0]))

                if self.attack_config["adaptive"]["delta_extraction"]:
                    cap = 1     # 每次最多选一个样本
                else:
                    cap = 1

                if np.sum(delta_good) >= cap:
                    x_advs = np.array(potential_advs)[
                        np.where(delta_good)[0]]  # TODO osd should be orig_satisfied only
                    x_advs_delta = x_advs.copy()
                    break
                elif np.sum(cache_preds[:,0])/len(cache_preds[:,0]) > 0.5:
                    if self.attack_config["adaptive"]["delta_extraction"]:
                        self.curr_delta /= self.step_adapt
            else:  # pragma: no cover
                return x_adv    # 所有尝试都失败了，返回当前最优对抗样本


            if self.curr_epsilon>1: self.curr_epsilon = initial_epsilon

            # Trust region method to adjust epsilon
            # Trust region：根据效果自适应调整 epsilon（向目标靠近的扰动大小）
            for _ in range(self.num_trial):
                # 向原始样本靠近
                perturb = np.repeat(np.array([original_sample]), len(x_advs), axis=0) - x_advs
                perturb *= self.curr_epsilon
                new_potential_advs = x_advs + perturb
                new_potential_advs = np.clip(new_potential_advs, clip_min, clip_max)
                # potential_advs = np.concatenate((x_advs, new_potential_advs))
                potential_advs = new_potential_advs

                # 预测新扰动样本
                output_preds, cache_preds = self.estimator.predict(potential_advs, batch_size=self.batch_size)
                preds = np.argmax(
                    output_preds,
                    axis=1,
                )

                # 若拒识样本不允许调整 epsilon，则直接返回
                if not self.attack_config["adaptive"]["eps_adjust_rejected_predictions"] and np.sum(np.where(cache_preds[:, 0], 1, 0)) > 0:
                    return "rejection in eps step"

                # 判断攻击是否成功
                if self.attack_config["targeted"]:
                    preds = np.where(cache_preds[:, 0], target, preds)
                else:
                    preds = np.where(cache_preds[:, 0], -1, preds)
                if self.attack_config["targeted"]:
                    is_adv = preds == target
                else:
                    is_adv = preds != y_p

                # 计算成功率
                if self.attack_config["adaptive"]["eps_extraction"]:
                    #print("is_adv", is_adv, "preds", preds)
                    epsilon_ratio = np.mean(is_adv)
                else:
                    # 如果全部样本被拒绝，尝试更新当前最优对抗样本
                    if cache_preds.shape[0] - np.sum(cache_preds[:, 0]) == 0:
                        if np.mean(delta_good) > 0:
                            x_adv = self._best_adv(original_sample, x_advs_delta)
                            self.curr_adv = x_adv
                            break
                        epsilon_ratio = 1
                    else:
                        epsilon_ratio = np.sum(is_adv[~np.array(cache_preds[:, 0])]) / (
                            cache_preds.shape[0] - np.sum(cache_preds[:, 0]))

                # 若全部拒识，尝试保存当前对抗样本
                if cache_preds.shape[0] == np.sum(cache_preds[:, 0]):
                    if np.mean(delta_good) > 0:
                        x_adv = self._best_adv(original_sample, x_advs_delta)
                        self.curr_adv = x_adv
                        break

                # if self.attack_config["adaptive"]["eps_extraction"]:
                #     #epsilon_ratio = np.sum(is_adv) / cache_preds.shape[0]
                #     if cache_preds.shape[0] - np.sum(cache_preds[:, 0]) == 0:
                #         #self.curr_epsilon /= (self.step_adapt/2)
                #         continue
                #     epsilon_ratio = np.sum(is_adv[~np.array(cache_preds[:, 0])]) / (cache_preds.shape[0] - np.sum(cache_preds[:, 0]))
                # else:
                #     if cache_preds.shape[0] - np.sum(cache_preds[:, 0]) == 0:
                #         continue
                #     epsilon_ratio = np.sum(is_adv[~np.array(cache_preds[:, 0])]) / (cache_preds.shape[0] - np.sum(cache_preds[:, 0]))

                # 判断是否有成功的扰动样本
                delta_good = is_adv * (preds >= 0)

                # if cache_preds.shape[0] == np.sum(cache_preds[:, 0]):
                #     if np.mean(delta_good) > 0:
                #         x_adv = self._best_adv(original_sample, x_advs_delta)
                #         self.curr_adv = x_adv
                #         break

                # print("epsilon_ratio: ", epsilon_ratio)
                # print("epsilon: ", self.curr_epsilon)
                # print(satisfied)

                #print(epsilon_ratio)
                # 更新 epsilon
                if epsilon_ratio < 0.2:
                    self.curr_epsilon *= self.step_adapt
                elif epsilon_ratio > 0.5:
                    self.curr_epsilon /= self.step_adapt
                # self.curr_epsilon = max(0.001, self.curr_epsilon)


                # print("condition satisfied: ", np.mean(orig_satisfied * non_rejects))
                if np.mean(delta_good) > 0:
                    x_adv = self._best_adv(original_sample, potential_advs[np.where(delta_good)[0]])
                    self.curr_adv = x_adv
                    break

            # 计算当前对抗样本的归一化 L2 距离
            l2_normalized = np.linalg.norm(x_adv - original_sample) / (
                    original_sample.shape[-1] * original_sample.shape[-2] * original_sample.shape[-3]) ** 0.5

            # 更新最优对抗样本
            if l2_normalized < best_l2:
                best_l2 = l2_normalized
                best_l2_set = this_iter

            # l_inf = np.max(np.abs(x_adv - original_sample))
            # 日志记录与进度显示
            log_msg = "Step : {} | L2 Normalized: {} | curr_epsilon: {}".format(this_iter, l2_normalized, self.curr_epsilon)
            pbar.set_description(log_msg)
            self.boundary_logger.info(log_msg)

            # 满足精度要求提前结束
            if l2_normalized < self.attack_config["eps"]:
                return x_adv
            elif self.curr_epsilon < 10e-6:
                return original_sample
        return x_adv    # 最终返回对抗样本

    def _orthogonal_perturb(self, delta: float, current_sample: np.ndarray, original_sample: np.ndarray) -> np.ndarray:
        """
        Create an orthogonal perturbation.

        :param delta: Initial step size for the orthogonal step.
        :param current_sample: Current adversarial example.
        :param original_sample: The original input.
        :return: a possible perturbation.
        """
        # Generate perturbation randomly
        perturb = np.random.randn(*self.estimator.input_shape).astype(ART_NUMPY_DTYPE)

        # Rescale the perturbation
        perturb /= np.linalg.norm(perturb)
        perturb *= delta * np.linalg.norm(original_sample - current_sample)

        # Project the perturbation onto sphere
        direction = original_sample - current_sample

        direction_flat = direction.flatten()
        perturb_flat = perturb.flatten()

        direction_flat /= np.linalg.norm(direction_flat)
        perturb_flat -= np.dot(perturb_flat, direction_flat.T) * direction_flat
        perturb = perturb_flat.reshape(self.estimator.input_shape)

        hypotenuse = np.sqrt(1 + delta ** 2)
        perturb = ((1 - hypotenuse) * (current_sample - original_sample) + perturb) / hypotenuse
        return perturb

    def _init_sample(
            self,
            x: np.ndarray,
            y: int,
            y_p: int,
            init_pred: int,
            adv_init: np.ndarray,
            clip_min: float,
            clip_max: float,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Find initial adversarial example for the attack.

        :param x: An array with one original input to be attacked.
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :param y_p: The predicted label of x.
        :param init_pred: The predicted label of the initial image.
        :param adv_init: Initial array to act as an initial adversarial example.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: an adversarial example.
        """
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            raise NotImplementedError
            # Attack satisfied
            if y == y_p:
                return None

            # Attack unsatisfied yet and the initial image satisfied
            if adv_init is not None and init_pred == y:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            # Attack unsatisfied yet and the initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                # test: either it's in cache mode OR it must be False
                output_preds, cache_preds = self.estimator.predict(np.array([random_img]), batch_size=self.batch_size)
                random_class = np.argmax(
                    output_preds,
                    axis=1,
                )[0]

                if not self.attack_config["adaptive"]["init_bypass_rejects"]:
                    return "rejected in initialization"
                if random_class == y and not cache_preds[0][0]:
                    initial_sample = random_img, random_class

                    logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            # The initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                class_preds, cache_preds = self.estimator.predict(np.array([random_img]), batch_size=self.batch_size)
                random_class = np.argmax(
                    class_preds,
                    axis=1,
                )[0]

                if not self.attack_config["adaptive"]["init_bypass_rejects"]:
                    return "rejected in initialization"
                # test: either it's in cache mode OR it must be False
                if random_class != y_p and not cache_preds[0][0]:
                    initial_sample = random_img, random_class

                    logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:  # pragma: no cover
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample

    @staticmethod
    def _best_adv(original_sample: np.ndarray, potential_advs: np.ndarray) -> np.ndarray:
        """
        From the potential adversarial examples, find the one that has the minimum L2 distance from the original sample

        :param original_sample: The original input.
        :param potential_advs: Array containing the potential adversarial examples
        :return: The adversarial example that has the minimum L2 distance from the original input
        """
        shape = potential_advs.shape
        min_idx = np.linalg.norm(original_sample.flatten() - potential_advs.reshape(shape[0], -1), axis=1).argmin()
        return potential_advs[min_idx]

    def _check_params(self) -> None:
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.num_trial, int) or self.num_trial < 0:
            raise ValueError("The number of trials must be a non-negative integer.")

        if not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise ValueError("The number of samples must be a positive integer.")

        if not isinstance(self.init_size, int) or self.init_size <= 0:
            raise ValueError("The number of initial trials must be a positive integer.")

        if self.epsilon <= 0:
            raise ValueError("The initial step size for the step towards the target must be positive.")

        if self.delta <= 0:
            raise ValueError("The initial step size for the orthogonal step must be positive.")

        if self.step_adapt <= 0 or self.step_adapt >= 1:
            raise ValueError("The adaptation factor must be in the range (0, 1).")

        if not isinstance(self.min_epsilon, (float, int)) or self.min_epsilon < 0:
            raise ValueError("The minimum epsilon must be non-negative.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
