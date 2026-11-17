# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metric definitions: key, default section, unit, definition, collection stage.

This registry describes fields supported by the presentation layer, not a promise
that every worker/loss produces them. Service fields require a validated reducer;
missing or inapplicable observations are never synthesized by this registry.
The keys and section labels below are the registry defaults. ``metric_catalog``
maps them to the numbered metric names actually logged to W&B.
"""

METRIC_DEFINITIONS = """\
pool/discarded_groups_total	Pool	组	本进程fully-async累计排除的组数，包含无效组、无训练target组和不足一批的尾部	Pool过滤
pool/skipped_groups_total	Pool	组	本进程累计因超过半数成员失败、跳过或reward无效而排除的完整组数	Pool过滤
pool/empty_groups_total	Pool	组	本进程累计没有有效预测target而排除的完整组数	Pool过滤
pool/tail_groups_total	Pool	组	本进程累计因数据源耗尽而丢弃的不足一批有效组数	Pool过滤
pool/discarded_trajectories_total	Pool	条	本进程被排除组内的全部父trajectory数	Pool过滤
pool/failed_trajectories_total	Pool	条	本进程被排除组内父状态为failed的trajectory数	Pool过滤
pool/skipped_trajectories_total	Pool	条	本进程被排除组内父状态为skipped的trajectory数	Pool过滤
pool/discarded_response_tokens_total	Pool	token	本进程被排除组内保存序列的response token数	Pool过滤
pool/discarded_rollout_seconds_total	Pool	秒	本进程被排除组内已报告的有限非负rollout耗时总和	Pool过滤
batch/group_count	Batch	组	本批交付的真实采样组数；非分组算法省略	取批汇总
batch/trajectory_count	Batch	条	本批交付的父trajectory总数	取批汇总
batch/segment_count	Batch	段	本批保存序列总数，包含被跳过组内的序列	取批汇总
batch/tokens_total	Batch	token	本批本地保存序列的完整token总数，包含response及非response位置；不含重采样和optimizer补齐复制，托管SampleRef不贡献本地token	取批汇总
batch/response_tokens_total	Batch	token	本批保存序列中loss_mask>0的位置总数	取批汇总
batch/skipped_group_count	Batch	组	被整组跳过的采样组数	取批汇总
batch/resampled_group_count	Batch	组	本批通过复制有效成员补齐的采样组数	训练准备
batch/resampled_trajectory_count	Batch	条	本批替换无效成员的trajectory副本数	训练准备
batch/failed_trajectory_count	Batch	条	父状态为failed的trajectory数	取批汇总
batch/skipped_trajectory_count	Batch	条	父状态为skipped的trajectory数	取批汇总
batch/empty_trajectory_count	Batch	条	没有保存序列的trajectory数	取批汇总
batch/trainable_group_count	Batch	组	至少贡献一个可训练segment的group数；不含optimizer补齐复制	Datum/mask处理后
batch/trainable_trajectory_count	Batch	条	至少贡献一个可训练segment的父trajectory数；不含optimizer补齐复制	Datum/mask处理后
batch/trainable_segment_count	Batch	段	具有有效response训练目标的原始segment数；不含optimizer补齐复制	Datum/mask处理后
batch/trainable_response_tokens_total	Batch	token	有效response训练token总数；不含optimizer补齐复制	Datum/mask处理后
batch/sandbox_retry_count	Batch	次	本批父trajectory的沙箱重试总次数；覆盖全部执行尝试，首次不计	执行采集→取批汇总
timeout/tool_count	Timeout	次	最终执行内已确认工具timeout事件数	执行采集→取批汇总
timeout/model_request_count	Timeout	次	最终执行内模型请求timeout事件数	执行采集→取批汇总
timeout/rollout_count	Timeout	条	最终整条执行超时的父trajectory数	执行采集→取批汇总
verifier/result_count	Verifier	条	提供可识别reward score或非空reason的父trajectory结果数，包含length跳过结果	判题采集→取批汇总
verifier/service_error_count	Verifier	条	最终reason为judge_error或timeout_after_retry的父trajectory数；不含普通答错、length和缺代码	判题采集→取批汇总
verifier/judge_error_count	Verifier	条	最终reason为judge_error或reasoning360/.../judge_error的父trajectory数；内部重试不另计	判题采集→取批汇总
verifier/timeout_count	Verifier	次	已报告的判题timeout_count之和；包含恢复成功的超时，只接受有限非负整数	判题采集→取批汇总
verifier/retry_count	Verifier	次	已报告的判题retry_count之和；一次CodeE2B结果可以包含多个测试batch重试	判题采集→取批汇总
verifier/retried_trajectory_count	Verifier	条	报告retry_count大于0或最终reason为timeout_after_retry的父trajectory数，每条只计一次	判题采集→取批汇总
verifier/retry_completed_count	Verifier	条	经历重试后返回普通判题结果的父trajectory数，包含判错给0，不含最终service_error	判题采集→取批汇总
verifier/timeout_after_retry_count	Verifier	条	重试耗尽后以timeout_after_retry给0的父trajectory数，不是超时事件数	判题采集→取批汇总
verifier/no_code_count	Verifier	条	提取不到非空代码而以no_code给0的父trajectory数，不计为服务失败	判题采集→取批汇总
reward_fn/<reward_fn>/verifier/<counter>	Reward Fn	条或次	指定有效reward_fn内上述verifier结果计数；父记录、事件次数与全batch使用相同口径	判题采集→取批汇总
reward_fn/<reward_fn>/reward_mean	Reward Fn	reward	本批指定非空reward_fn内已知有限最终reward的父trajectory等权均值；有效0参与，缺失或非有限值不参与	取批汇总
reward_fn/<reward_fn>/reward_min	Reward Fn	reward	本批指定非空reward_fn内已知有限最终reward的最小值；无有效观测时省略	取批汇总
reward_fn/<reward_fn>/reward_max	Reward Fn	reward	本批指定非空reward_fn内已知有限最终reward的最大值；无有效观测时省略	取批汇总
reward_fn/<reward_fn>/reward_std	Reward Fn	reward	本批指定非空reward_fn内已知有限最终reward的总体标准差；每个父trajectory等权，无有效观测时省略	取批汇总
reward_fn/<reward_fn>/reward_observed_count	Reward Fn	条	本批指定非空reward_fn内参与reward统计的父trajectory数；每个父对象计一次，全无有效reward时为0	取批汇总
val/verifier/<counter>	Val	条或次	本次验证上述verifier结果计数；仅统计已收到的父trajectory结果，不累计历史验证	V
val/reward_fn/<reward_fn>/verifier/<counter>	Val	条或次	本次验证指定有效reward_fn内上述verifier结果计数	V
algorithm/trajectory/reward_mean	Algorithm	reward	已知有限最终reward的平均值	取批汇总
algorithm/trajectory/reward_min	Algorithm	reward	已知有限最终reward的最小值	取批汇总
algorithm/trajectory/reward_max	Algorithm	reward	已知有限最终reward的最大值	取批汇总
algorithm/trajectory/reward_std	Algorithm	reward	已知有限最终reward的总体标准差	取批汇总
algorithm/trajectory/reward_observed_count	Algorithm	条	实际参与reward统计的父trajectory数，每个父对象计一次	取批汇总
algorithm/group/reward_std_min	Algorithm	reward	各有效组内部reward样本标准差的最小值	GRPO准备
algorithm/group/reward_std_max	Algorithm	reward	各有效组内部reward样本标准差的最大值	GRPO准备
algorithm/trajectory/advantage_mean	Algorithm	advantage	有效GRPO父trajectory的advantage的平均值	GRPO准备
algorithm/trajectory/advantage_min	Algorithm	advantage	有效GRPO父trajectory的advantage的最小值	GRPO准备
algorithm/trajectory/advantage_max	Algorithm	advantage	有效GRPO父trajectory的advantage的最大值	GRPO准备
algorithm/trajectory/advantage_std	Algorithm	advantage	有效GRPO父trajectory的advantage的总体标准差	GRPO准备
algorithm/trajectory/advantage_abs_mean	Algorithm	advantage	有效GRPO父trajectory的advantage的绝对值的平均值	GRPO准备
algorithm/trajectory/advantage_token_weighted_mean	Algorithm	advantage	按各父trajectory保存response token量加权的advantage均值	GRPO准备
algorithm/trajectory/advantage_positive_count	Algorithm	条	advantage大于0的有效GRPO父trajectory数	GRPO准备
algorithm/trajectory/advantage_negative_count	Algorithm	条	advantage小于0的有效GRPO父trajectory数	GRPO准备
algorithm/trajectory/advantage_zero_count	Algorithm	条	advantage等于0的有效GRPO父trajectory数	GRPO准备
algorithm/trajectory/advantage_zero_percentage	Algorithm	%	零advantage数/有效advantage父trajectory数×100	GRPO准备
algorithm/kl/actor_ref_mean	Algorithm	KL估计	实际训练forward中target对齐有效response位置的actor/reference配置KL估计，未乘coef；跨更新按token加权，含服务实际收到的补齐或重复样本	训练forward→本轮汇总
algorithm/kl/actor_ref_loss	Algorithm	loss贡献	服务明确返回的Reference KL实际loss贡献，按真实loss归一化汇总；不使用coefficient×mean推算	训练forward→本轮汇总
algorithm/kl/coefficient	Algorithm	系数	本step的reference KL系数	训练forward→本轮汇总
algorithm/weaver/entropy_coeff	Algorithm	系数	对应loss实际使用的熵系数	服务返回→本轮汇总
algorithm/weaver/entropy_loss	Algorithm	loss	服务熵项，符号/系数/归一化待确认	服务返回→本轮汇总
algorithm/weaver/entropy	Algorithm	nats	温度1的全词表策略熵；有效loss mask token的entropy:sum/entropy:tokens，跨请求和更新合并分子分母；与entropy_coeff和loss_agg_mode无关；服务明确返回空mask时为0	训练forward→本轮汇总
algorithm/weaver/entropy:sum	Algorithm	nats	有效loss mask token的全词表策略熵总和；跨请求和更新求和	训练forward→本轮汇总
algorithm/weaver/entropy:tokens	Algorithm	token	全词表策略熵统计的有效loss mask token数；跨请求和更新求和，排除被mask的prompt、padding和filler	训练forward→本轮汇总
algorithm/weaver/gspo/ratio_mean	Algorithm	无量纲	GSPO长度归一化序列概率比均值；跨optimizer batch按非空完整trajectory数加权	GSPO callback→本轮汇总
algorithm/weaver/gspo/ratio_min	Algorithm	无量纲	GSPO非空完整trajectory概率比最小值；跨optimizer batch取最小值	GSPO callback→本轮汇总
algorithm/weaver/gspo/ratio_max	Algorithm	无量纲	GSPO非空完整trajectory概率比最大值；跨optimizer batch取最大值	GSPO callback→本轮汇总
algorithm/weaver/gspo/clip_fraction	Algorithm	0–1	GSPO目标裁剪生效的非空完整trajectory占比；跨optimizer batch按非空完整trajectory数加权	GSPO callback→本轮汇总
algorithm/weaver/gspo/active_trajectories	Algorithm	序列	GSPO参与序列均值的非空完整trajectory数，包含零advantage；跨optimizer batch求和	GSPO callback→本轮汇总
algorithm/weaver/gspo/empty_trajectories	Algorithm	序列	GSPO callback收到的无response token完整trajectory数；跨optimizer batch求和，全空且跳过的batch不计入	GSPO callback→本轮汇总
algorithm/weaver/ppo_clip/clipfrac	Algorithm	比例，精确分母待确认	PPO裁剪比例；需确认是比率越界还是目标裁剪生效比例	训练forward→本轮汇总
algorithm/weaver/ppo_clip/clipfrac_high	Algorithm	比例，精确分母待确认	服务定义的high侧裁剪统计，条件及分母需确认	训练forward→本轮汇总
algorithm/weaver/ppo_clip/clipfrac_low	Algorithm	比例，精确分母待确认	服务定义的low侧裁剪统计，条件及分母需确认	训练forward→本轮汇总
algorithm/weaver/ppo_clip/ppo_kl	Algorithm	KL估计	当前训练策略相对固定分母策略的KL估计；确认与负is:logratio等价时只展示本字段	训练forward→本轮汇总
algorithm/weaver/is:ratio	Algorithm	无量纲	概率比r的服务统计值；均值口径需确认	训练forward→本轮汇总
algorithm/weaver/is:ratio_min	Algorithm	无量纲	概率比r的最小值	训练forward→本轮汇总
algorithm/weaver/is:ratio_max	Algorithm	无量纲	概率比r的最大值	训练forward→本轮汇总
algorithm/weaver/is:ratio_sq	Algorithm	无量纲	概率比r平方的统计，不能解释成最终均值再平方	训练forward→本轮汇总
algorithm/weaver/is:logratio	Algorithm	无量纲	log(r)的服务统计值；与ppo_kl正负号等价时不重复展示	训练forward→本轮汇总
algorithm/weaver/is:logratio_min	Algorithm	无量纲	log(r)最小值	训练forward→本轮汇总
algorithm/weaver/is:logratio_max	Algorithm	无量纲	log(r)最大值	训练forward→本轮汇总
algorithm/weaver/is:logratio_sq	Algorithm	无量纲	log(r)平方的统计	训练forward→本轮汇总
algorithm/weaver/is:logratio_std	Algorithm	无量纲	log(r)标准差	训练forward→本轮汇总
algorithm/weaver/is:abs_logratio	Algorithm	无量纲	绝对log(r)的服务统计值	训练forward→本轮汇总
algorithm/weaver/is:abs_logratio_max	Algorithm	无量纲	绝对log(r)最大值	训练forward→本轮汇总
algorithm/weaver/ratio:mean	Algorithm	无量纲	有效response token上概率比平均值；truncated自定义loss下可能已截断	训练forward→本轮汇总
algorithm/weaver/ratio:min	Algorithm	无量纲	有效response token上概率比最小值；truncated自定义loss下可能已截断	训练forward→本轮汇总
algorithm/weaver/ratio:max	Algorithm	无量纲	有效response token上概率比最大值；truncated自定义loss下可能已截断	训练forward→本轮汇总
algorithm/weaver/ratio:std	Algorithm	无量纲	有效response token上概率比总体标准差；truncated自定义loss下可能已截断	训练forward→本轮汇总
algorithm/weaver/ratio:gt1_frac	Algorithm	0–1	有效response token中概率比大于1的比例；truncated自定义loss下可能已截断	训练forward→本轮汇总
algorithm/weaver/ratio:gt2_frac	Algorithm	0–1	有效response token中概率比大于2的比例；truncated自定义loss下可能已截断	训练forward→本轮汇总
algorithm/weaver/ratio:gt5_frac	Algorithm	0–1	有效response token中概率比大于5的比例；truncated自定义loss下可能已截断	训练forward→本轮汇总
algorithm/weaver/logprob_diff:mean	Algorithm	log概率差	当前训练logprob减分母logprob的平均值	训练forward→本轮汇总
algorithm/weaver/logprob_diff:min	Algorithm	log概率差	当前训练logprob减分母logprob的最小值	训练forward→本轮汇总
algorithm/weaver/logprob_diff:max	Algorithm	log概率差	当前训练logprob减分母logprob的最大值	训练forward→本轮汇总
algorithm/weaver/logprob_diff:std	Algorithm	log概率差	当前训练logprob减分母logprob的总体标准差	训练forward→本轮汇总
trajectory/response_tokens_mean	Trajectory	token	每条trajectory保存response token总量的平均值	取批汇总
trajectory/response_tokens_min	Trajectory	token	每条trajectory保存response token总量的最小值	取批汇总
trajectory/response_tokens_max	Trajectory	token	每条trajectory保存response token总量的最大值	取批汇总
trajectory/segments_mean	Trajectory	段	每条trajectory保存segment数的平均值	取批汇总
trajectory/segments_min	Trajectory	段	每条trajectory保存segment数的最小值	取批汇总
trajectory/segments_max	Trajectory	段	每条trajectory保存segment数的最大值	取批汇总
trajectory/finish_reason_length_count	Trajectory	条	本批父trajectory的finish_reason为length的数量；每条父记录只计一次	取批汇总
trajectory/finish_reason_length_fraction	Trajectory	0–1	本批finish_reason为length的父trajectory数/全部父trajectory数；缺失finish_reason按非length计	取批汇总
trajectory/model_requests_mean	Trajectory	次	每条trajectory最终执行内模型请求数的平均值	执行采集→取批汇总
trajectory/model_requests_min	Trajectory	次	每条trajectory最终执行内模型请求数的最小值	执行采集→取批汇总
trajectory/model_requests_max	Trajectory	次	每条trajectory最终执行内模型请求数的最大值	执行采集→取批汇总
trajectory/tool_calls_mean	Trajectory	次	每条trajectory最终执行内工具调用数的平均值	执行采集→取批汇总
trajectory/tool_calls_min	Trajectory	次	每条trajectory最终执行内工具调用数的最小值	执行采集→取批汇总
trajectory/tool_calls_max	Trajectory	次	每条trajectory最终执行内工具调用数的最大值	执行采集→取批汇总
trajectory/tool_timeouts_mean	Trajectory	次	每条trajectory最终执行内确认工具timeout数的平均值	执行采集→取批汇总
trajectory/tool_timeouts_min	Trajectory	次	每条trajectory最终执行内确认工具timeout数的最小值	执行采集→取批汇总
trajectory/tool_timeouts_max	Trajectory	次	每条trajectory最终执行内确认工具timeout数的最大值	执行采集→取批汇总
trajectory/model_request_timeouts_mean	Trajectory	次	每条trajectory最终执行内模型请求timeout数的平均值	执行采集→取批汇总
trajectory/model_request_timeouts_min	Trajectory	次	每条trajectory最终执行内模型请求timeout数的最小值	执行采集→取批汇总
trajectory/model_request_timeouts_max	Trajectory	次	每条trajectory最终执行内模型请求timeout数的最大值	执行采集→取批汇总
trajectory/sandbox_retries_mean	Trajectory	次	每条trajectory全部沙箱重试次数的平均值	执行采集→取批汇总
trajectory/sandbox_retries_min	Trajectory	次	每条trajectory全部沙箱重试次数的最小值	执行采集→取批汇总
trajectory/sandbox_retries_max	Trajectory	次	每条trajectory全部沙箱重试次数的最大值	执行采集→取批汇总
timing_rollout/trajectory_mean_seconds	Timing Rollout	秒	每条trajectory从首次执行开始至最终结果返回的完整耗时的平均值	执行采集→取批汇总
timing_rollout/trajectory_min_seconds	Timing Rollout	秒	每条trajectory从首次执行开始至最终结果返回的完整耗时的最小值	执行采集→取批汇总
timing_rollout/trajectory_max_seconds	Timing Rollout	秒	每条trajectory从首次执行开始至最终结果返回的完整耗时的最大值	执行采集→取批汇总
timing_rollout/verifier_mean_seconds	Timing Rollout	秒	实际reward_fn逻辑调用的完整耗时均值，含内部重试和沙箱清理；length跳过不参与	执行采集→取批汇总
timing_rollout/verifier_min_seconds	Timing Rollout	秒	实际reward_fn逻辑调用的完整耗时最小值	执行采集→取批汇总
timing_rollout/verifier_max_seconds	Timing Rollout	秒	实际reward_fn逻辑调用的完整耗时最大值	执行采集→取批汇总
verifier/skipped_length_count	Verifier	条	明确因finish_reason为length而跳过verifier的父trajectory数	执行采集→取批汇总
timing_rollout/<reward_fn>_trajectory_mean_seconds	Timing Rollout	秒	指定reward_fn的父trajectory完整耗时均值，与全batch使用同一口径	执行采集→取批汇总
timing_rollout/<reward_fn>_trajectory_min_seconds	Timing Rollout	秒	指定reward_fn的父trajectory完整耗时最小值	执行采集→取批汇总
timing_rollout/<reward_fn>_trajectory_max_seconds	Timing Rollout	秒	指定reward_fn的父trajectory完整耗时最大值	执行采集→取批汇总
timing_rollout/<reward_fn>_group_mean_seconds	Timing Rollout	秒	指定reward_fn的完整采样组执行跨度均值；每个组等权	执行采集→取批汇总
timing_rollout/<reward_fn>_group_min_seconds	Timing Rollout	秒	指定reward_fn的完整采样组执行跨度最小值	执行采集→取批汇总
timing_rollout/<reward_fn>_group_max_seconds	Timing Rollout	秒	指定reward_fn的完整采样组执行跨度最大值	执行采集→取批汇总
timing_rollout/<reward_fn>_verifier_mean_seconds	Timing Rollout	秒	指定reward_fn实际判题调用的完整耗时均值；length跳过不参与	执行采集→取批汇总
timing_rollout/<reward_fn>_verifier_min_seconds	Timing Rollout	秒	指定reward_fn实际判题调用的完整耗时最小值；length跳过不参与	执行采集→取批汇总
timing_rollout/<reward_fn>_verifier_max_seconds	Timing Rollout	秒	指定reward_fn实际判题调用的完整耗时最大值；length跳过不参与	执行采集→取批汇总
reward_fn/<reward_fn>/verifier/skipped_length_count	Reward Fn	条	指定reward_fn内明确因length跳过判题的父trajectory数	执行采集→取批汇总
segment/response_tokens_mean	Segment	token	每个保存segment的response token数的平均值	取批汇总
segment/response_tokens_min	Segment	token	每个保存segment的response token数的最小值	取批汇总
segment/response_tokens_max	Segment	token	每个保存segment的response token数的最大值	取批汇总
segment/tool_calls_mean	Segment	次	每段可可靠归属的工具调用数的平均值	执行采集→取批汇总
segment/tool_calls_min	Segment	次	每段可可靠归属的工具调用数的最小值	执行采集→取批汇总
segment/tool_calls_max	Segment	次	每段可可靠归属的工具调用数的最大值	执行采集→取批汇总
segment/tool_timeouts_mean	Segment	次	每段可可靠归属的确认工具timeout数的平均值	执行采集→取批汇总
segment/tool_timeouts_min	Segment	次	每段可可靠归属的确认工具timeout数的最小值	执行采集→取批汇总
segment/tool_timeouts_max	Segment	次	每段可可靠归属的确认工具timeout数的最大值	执行采集→取批汇总
segment/model_request_timeouts_mean	Segment	次	每段可可靠归属的模型请求timeout数的平均值	执行采集→取批汇总
segment/model_request_timeouts_min	Segment	次	每段可可靠归属的模型请求timeout数的最小值	执行采集→取批汇总
segment/model_request_timeouts_max	Segment	次	每段可可靠归属的模型请求timeout数的最大值	执行采集→取批汇总
segment/truncated_count	Segment	段	实际发生token裁剪的保存segment数	取批汇总
segment/truncated_percentage	Segment	%	实际裁剪段数/本批所有保存段数×100；仅全部段有明确裁剪观测且分母>0时记录	取批汇总
group/response_tokens_min	Group	token	各组response token总量的最小值	取批汇总
group/response_tokens_max	Group	token	各组response token总量的最大值	取批汇总
group/segments_min	Group	段	各组segment总数的最小值	取批汇总
group/segments_max	Group	段	各组segment总数的最大值	取批汇总
group/trajectories_min	Group	条	各组父trajectory数的最小值	取批汇总
group/trajectories_max	Group	条	各组父trajectory数的最大值	取批汇总
group/model_requests_min	Group	次	各组最终执行内模型请求总次数的最小值	执行采集→取批汇总
group/model_requests_max	Group	次	各组最终执行内模型请求总次数的最大值	执行采集→取批汇总
group/tool_calls_min	Group	次	各组最终执行内工具调用总次数的最小值	执行采集→取批汇总
group/tool_calls_max	Group	次	各组最终执行内工具调用总次数的最大值	执行采集→取批汇总
group/tool_timeouts_min	Group	次	各组最终执行内确认工具timeout总次数的最小值	执行采集→取批汇总
group/tool_timeouts_max	Group	次	各组最终执行内确认工具timeout总次数的最大值	执行采集→取批汇总
group/model_request_timeouts_min	Group	次	各组最终执行内模型请求timeout总次数的最小值	执行采集→取批汇总
group/model_request_timeouts_max	Group	次	各组最终执行内模型请求timeout总次数的最大值	执行采集→取批汇总
group/sandbox_retries_min	Group	次	各组父trajectory全部沙箱重试总次数的最小值	执行采集→取批汇总
group/sandbox_retries_max	Group	次	各组父trajectory全部沙箱重试总次数的最大值	执行采集→取批汇总
timing_rollout/group_mean_seconds	Timing Rollout	秒	完整采样组从最早worker开始到最晚结果完成的时间跨度均值；每个组等权	执行采集→取批汇总
timing_rollout/group_min_seconds	Timing Rollout	秒	完整采样组执行时间跨度的最小值	执行采集→取批汇总
timing_rollout/group_max_seconds	Timing Rollout	秒	完整采样组执行时间跨度的最大值	执行采集→取批汇总
group/valid_group_count	Group	组	参与GRPO基线/advantage计算的有效组数，可含零训练token组	GRPO准备
group/singleton_group_count	Group	组	有效组中只有一个成员的组数	GRPO准备
training/train_step	Training	步	已完成的batch处理轮次；唯一横坐标	train返回
training/optimizer_step_count	Training	次	本轮实际完成的optimizer更新次数	训练完成
training/forward_backward_call_count	Training	次	本轮实际forward/backward请求次数	训练完成
training/skipped_flag	Training	0/1	本轮是否跳过参数更新，1为跳过	训练完成
training/weight_export_success_flag	Training	0/1	本轮权重导出是否成功；未执行则省略	导出结束
training/weight_sync_success_flag	Training	0/1	本轮采样权重同步是否成功；未执行则省略	同步结束
weaver/loss	Training	loss	本轮训练目标值，按真实loss reduction正确汇总	训练forward→本轮汇总
weaver/grad_norm	Training	梯度范数	每次成功optimizer更新返回的梯度范数；多更新按观测值等权平均，裁剪前后由服务定义	optimizer返回
model_request/input_tokens_mean	Model request	token	单次请求完整输入长度，含历史和工具结果的平均值	执行采集→取批汇总
model_request/input_tokens_min	Model request	token	单次请求完整输入长度，含历史和工具结果的最小值	执行采集→取批汇总
model_request/input_tokens_max	Model request	token	单次请求完整输入长度，含历史和工具结果的最大值	执行采集→取批汇总
model_request/output_tokens_mean	Model request	token	单次请求新生成输出长度的平均值	执行采集→取批汇总
model_request/output_tokens_min	Model request	token	单次请求新生成输出长度的最小值	执行采集→取批汇总
model_request/output_tokens_max	Model request	token	单次请求新生成输出长度的最大值	执行采集→取批汇总
model_request/total_tokens_mean	Model request	token	每次请求input+output之和的平均值	执行采集→取批汇总
model_request/total_tokens_min	Model request	token	每次请求input+output之和的最小值	执行采集→取批汇总
model_request/total_tokens_max	Model request	token	每次请求input+output之和的最大值	执行采集→取批汇总
timing_rollout/model_request_mean_seconds	Timing Rollout	秒	单次generate调用耗时，含内部等待/传输/重试，不含返回后的转换录制的平均值	执行采集→取批汇总
timing_rollout/model_request_min_seconds	Timing Rollout	秒	单次generate调用耗时，含内部等待/传输/重试，不含返回后的转换录制的最小值	执行采集→取批汇总
timing_rollout/model_request_max_seconds	Timing Rollout	秒	单次generate调用耗时，含内部等待/传输/重试，不含返回后的转换录制的最大值	执行采集→取批汇总
model_request/count	Model request	次	最终执行内发起模型请求总数	执行采集→取批汇总
model_request/completed_count	Model request	次	已结束请求总数，含成功/错误/timeout/取消	执行采集→取批汇总
model_request/token_usage_observed_count	Model request	次	有有效input/output token观测的请求数	执行采集→取批汇总
tool/calls_emitted_count	Tool	次	最终执行内发出的工具调用总数	执行采集→取批汇总
tool/results_observed_count	Tool	个	按调用身份去重的已观测工具结果数	执行采集→取批汇总
tool/result_error_count	Tool	个	已确认失败的工具结果数	执行采集→取批汇总
tool/unobserved_call_count	Tool	次	发出后没有观测到结果的调用数，不等于timeout	执行采集→取批汇总
tool/unassigned_result_count	Tool	个	结果已观测到但不能关联发起segment的数量	执行采集→取批汇总
tool/error_command_not_found_count	Tool	个	缺命令或可执行文件类失败结果数	执行采集→取批汇总
tool/error_dependency_missing_count	Tool	个	缺依赖类失败结果数	执行采集→取批汇总
tool/error_network_count	Tool	个	网络类失败结果数	执行采集→取批汇总
tool/error_permission_count	Tool	个	权限类失败结果数	执行采集→取批汇总
tool/error_resource_count	Tool	个	资源类失败结果数	执行采集→取批汇总
tool/error_file_not_found_count	Tool	个	缺文件类失败结果数	执行采集→取批汇总
tool/error_edit_count	Tool	个	编辑/补丁类失败结果数	执行采集→取批汇总
tool/error_test_or_compile_count	Tool	个	测试/编译类失败结果数	执行采集→取批汇总
tool/error_other_count	Tool	个	其他类失败结果数	执行采集→取批汇总
tool/calls_detected_count	Tool	次	已知数量的检测调用总数；NorthGate未知数量不补估计值	执行采集→取批汇总
tool/calls_invalid_count	Tool	次	已知数量的解析或结构无效调用总数；NorthGate原文计数不与返回调用重复累加	执行采集→取批汇总
tool/format_error_turn_count	Tool	条回复	明确有工具格式错误的模型回复数	执行采集→取批汇总
tool/format_unknown_turn_count	Tool	条回复	无法确认工具格式有效性的回复数	执行采集→取批汇总
tool/call_counts_unknown_turn_count	Tool	条回复	NorthGate诊断中detected或invalid数量未知的回复数	执行采集→取批汇总
tool/format_reason_incomplete_tool_call_turn_count	Tool	条回复	NorthGate报告工具调用未闭合的回复数，同轮只计一次	执行采集→取批汇总
tool/format_reason_invalid_arguments_json_turn_count	Tool	条回复	NorthGate报告参数JSON错误的回复数，同轮只计一次	执行采集→取批汇总
tool/format_reason_missing_tool_name_turn_count	Tool	条回复	NorthGate报告缺少工具名的回复数，同轮只计一次	执行采集→取批汇总
tool/format_reason_invalid_tool_call_structure_turn_count	Tool	条回复	NorthGate报告工具调用结构错误的回复数，同轮只计一次	执行采集→取批汇总
tool/format_reason_arguments_not_object_turn_count	Tool	条回复	NorthGate报告参数不是对象的回复数，同轮只计一次	执行采集→取批汇总
tool/format_reason_duplicate_parameter_turn_count	Tool	条回复	NorthGate报告重复参数的回复数，同轮只计一次	执行采集→取批汇总
tool/format_reason_undeclared_tool_turn_count	Tool	条回复	NorthGate报告未声明工具名的回复数，同轮只计一次	执行采集→取批汇总
tool/arguments_json_error_count	Tool	次	参数JSON解析失败次数	执行采集→取批汇总
tool/arguments_type_error_count	Tool	次	JSON合法但参数不是对象的调用数	执行采集→取批汇总
timeout/trajectory_with_tool_timeout_count	Timeout	条	最终执行至少一次工具timeout的父trajectory数，去重	执行采集→取批汇总
timeout/segment_with_tool_timeout_count	Timeout	段	至少一次可归属工具timeout的保存段数，去重	执行采集→取批汇总
timeout/trajectory_with_model_request_timeout_count	Timeout	条	最终执行至少一次模型请求timeout的父trajectory数	执行采集→取批汇总
timeout/segment_with_model_request_timeout_count	Timeout	段	至少一次可归属模型请求timeout的保存段数	执行采集→取批汇总
timeout/segment_from_timed_out_rollout_count	Timeout	段	最终超时父trajectory携带的保存段数，不是段自身timeout次数	执行采集→取批汇总
outcome/env_error_count	Outcome	条	明确环境或基础设施错误的父结果数	取批汇总
outcome/task_failed_count	Outcome	条	有效执行后的任务失败数，区别于执行异常	取批汇总
outcome/context_limit_reached_count	Outcome	条	明确因context_length_exceeded结束的父trajectory数	取批汇总
outcome/truncated_trajectory_count	Outcome	条	至少一个保存段实际裁剪的父trajectory数	取批汇总
outcome/resolved_count	Outcome	条	有明确SWE评估且resolved的trajectory数	取批汇总
outcome/evaluated_trajectory_count	Outcome	条	有明确SWE评估结果的trajectory数	取批汇总
outcome/resolved_percentage	Outcome	%	resolved/evaluated×100；仅SWE适用且分母>0时发布	取批汇总
outcome/score_<name>_mean	Outcome	分数	已注册附加任务分数的有效父trajectory等权均值	取批汇总
outcome/score_<name>_observed_count	Outcome	条	对应附加分数的有效观测数量	取批汇总
algorithm/sampling/logprob_mean	Algorithm	log概率	有有效观测的response token原采样logprob的均值，按token统计	GRPO准备
algorithm/sampling/logprob_min	Algorithm	log概率	有有效观测的response token原采样logprob的最小值，按token统计	GRPO准备
algorithm/sampling/logprob_max	Algorithm	log概率	有有效观测的response token原采样logprob的最大值，按token统计	GRPO准备
algorithm/sampling/logprob_std	Algorithm	log概率	有有效观测的response token原采样logprob的总体标准差，按token统计	GRPO准备
algorithm/sampling/ppl_mean	Algorithm	PPL	先逐segment算exp(-平均response logprob)，再按segment等权平均	GRPO准备
weaver/update_successful	Weaver	标记	服务更新成功标记；同义于客户端成功更新信息时不重复曲线	服务返回→本轮汇总
weaver/tokens	Weaver	token	服务/自定义loss token计量；可能含补齐/重复，不等于batch唯一token数	服务返回→本轮汇总
weaver/num_samples	Weaver	Datum/样本	服务样本计量，不能叫父trajectory数	服务返回→本轮汇总
timing/batch_completion_interval_seconds	Timing	秒	相邻两次train返回之间的墙钟间隔，含上一轮同步/验证/等待及本轮训练与保存；首个观测轮次省略	train返回
timing/trainer_wait_for_batch_seconds	Timing	秒	trainer因输入batch尚未就绪等待的累计时间，含轮询间隔；立即领取为0，不含训练、权重同步或验证	取批
timing_train/batch_training_seconds	Timing Train	秒	进入标准train到保存前的墙钟耗时，含数据处理、重算及FB/optimizer，不含权重导出、checkpoint或同步；自定义trainer未分段时回退完整调用	训练阶段结束
timing_train/recompute_logprobs_seconds	Timing Train	秒	训练端重算logprob的客户端墙钟耗时，含请求等待	重算结束
timing_train/ref_logprobs_seconds	Timing Train	秒	reference模型前向计算logprob的客户端墙钟耗时	reference前向结束
timing_train/mismatch_forward_seconds	Timing Train	秒	为差异诊断额外进行训练端前向的客户端墙钟耗时，不含随后统计	前向结束
timing_train/forward_backward_wall_seconds	Timing Train	秒	本步FB阶段客户端墙钟耗时合计，含数据准备、提交、排队与结果轮询；流水线按整体阶段跨度计	全部FB完成
timing_train/forward_backward_compute_seconds	Timing Train	秒	本步FB服务返回compute_ms的完整有效观测合计除以1000；服务计算口径，不等于客户端等待或纯GPU时间	全部FB返回
timing_train/optimizer_step_wall_seconds	Timing Train	秒	本步optimizer提交到结果完成的客户端墙钟耗时合计；在全部FB成功后开始	optimizer返回
timing_train/optimizer_step_compute_seconds	Timing Train	秒	本步optimizer服务返回compute_ms的完整有效观测合计除以1000	optimizer返回
timing_weights/weight_export_seconds	Timing Weights	秒	采样权重导出成功路径客户端墙钟耗时，不计入训练总耗时	导出结束
timing_weights/checkpoint_seconds	Timing Weights	秒	本步checkpoint保存成功路径客户端墙钟耗时，不计入训练总耗时	checkpoint结束
timing_weights/weight_sync_seconds	Timing Weights	秒	train返回后的采样权重同步客户端墙钟耗时	同步结束
timing_weights/weight_export_failed_seconds	Timing Weights	秒	导出失败前耗时	失败路径
timing_weights/weight_sync_failed_seconds	Timing Weights	秒	同步失败前耗时	失败路径
val/trajectory_count	Val	条	本次验证收到的父trajectory记录数；包含失败结果，普通总量统计尚未按run_id去重	V
timing_val/trajectory_mean_seconds	Timing Val	秒	本次验证有有效计时的父trajectory完整耗时均值	V
timing_val/trajectory_min_seconds	Timing Val	秒	本次验证有有效计时的父trajectory完整耗时最小值	V
timing_val/trajectory_max_seconds	Timing Val	秒	本次验证有有效计时的父trajectory完整耗时最大值	V
timing_val/group_mean_seconds	Timing Val	秒	本次验证完整采样组执行跨度均值；每个组等权	V
timing_val/group_min_seconds	Timing Val	秒	本次验证完整采样组执行跨度最小值	V
timing_val/group_max_seconds	Timing Val	秒	本次验证完整采样组执行跨度最大值	V
timing_val/verifier_mean_seconds	Timing Val	秒	本次验证实际reward_fn调用的完整耗时均值；length跳过不参与	V
timing_val/verifier_min_seconds	Timing Val	秒	本次验证实际reward_fn调用的完整耗时最小值；length跳过不参与	V
timing_val/verifier_max_seconds	Timing Val	秒	本次验证实际reward_fn调用的完整耗时最大值；length跳过不参与	V
val/verifier/skipped_length_count	Val	条	本次验证明确因length跳过判题的父trajectory数	V
timing_val/<reward_fn>_<object>_mean_seconds	Timing Val	秒	本次验证指定reward_fn的trajectory、group或verifier耗时均值	V
timing_val/<reward_fn>_<object>_min_seconds	Timing Val	秒	本次验证指定reward_fn的trajectory、group或verifier耗时最小值	V
timing_val/<reward_fn>_<object>_max_seconds	Timing Val	秒	本次验证指定reward_fn的trajectory、group或verifier耗时最大值	V
val/finish_reason_length_count	Val	条	本次验证父trajectory的finish_reason为length的数量；与trajectory_count同口径，未按run_id去重	V
val/finish_reason_length_fraction	Val	0–1	本次验证finish_reason为length的父trajectory数/全部父trajectory数；缺失finish_reason按非length计	V
val/<dataset>/trajectory_count	Val	条	指定验证数据集收到的父trajectory记录数；包含失败结果，未按run_id去重	V
val/<dataset>/finish_reason_length_count	Val	条	指定验证数据集父trajectory的finish_reason为length的数量；每条父记录只计一次	V
val/<dataset>/finish_reason_length_fraction	Val	0–1	指定验证数据集finish_reason为length的父trajectory数/该数据集全部父trajectory数；缺失finish_reason按非length计	V
val/segment_count	Val	段	本次验证父结果保存的训练序列/segment总数；普通有tokens的父对象贡献1段，空结果0段	V
val/<score>_mean	Val	由score定义	跨本次验证所有数据集，对提供该有限数值分数的父结果等权求均值；不是各数据集等权平均	V
val/<score>_observed_count	Val	条	参与上述分数均值的父结果数，每个score使用自己的分母	V
val/<dataset>/<score>_mean	Val	由score定义	指定验证数据集内，对该score有效观测等权求均值	V
val/<dataset>/<score>_observed_count	Val	条	参与指定数据集该score均值的父结果数	V
val/<dataset>/pass@1	Val	0–1	每题前k个去重run槽位的正确数c除以k，再对该数据集题目等权平均；不足k的缺失槽位按错误处理	V
val/<dataset>/pass@<k>	Val	0–1	每题前k个去重run中是否至少一次正确，再对该数据集题目等权平均	V
val/<dataset>/num_prompts	Val	题	实际观察到的不同题目组数；完全没有任何结果的计划题目不会进入此数	V
val/<dataset>/num_generations	Val	次	每题按run_id去重排序后选取前k条的实际观测生成数之和，每题最多k条，不含补0槽位	V
val/<dataset>/observed_generations/mean	Val	次/题	上述实际观测生成数除以已观察题目组数；每题最多k条	V
val/<dataset>/complete_group_ratio	Val	0–1	至少观察到k个不同run的题目组数/已观察题目组数；不是整个计划验证集的任务完成率	V
val/k	Val	次/题	本次验证配置的每题重复采样次数	V
val/duplicate_run_count	Val	条	同一dataset、同一group中重复run_id记录的数量，保留第一条，后续每条重复各加1	V
algorithm/mis/metrics_available	Algorithm	0/1	概率不匹配诊断成功且有至少一个有效token为1，否则0；不表示整个MIS功能是否可用	M
algorithm/mis/original_tokens	Algorithm	token	固定MIS处理前loss_mask=1的token总数	M
algorithm/mis/kept_tokens	Algorithm	token	TIS mask、RS、veto之后correction mask仍为1的token总数	M
algorithm/mis/original_sequences	Algorithm	段	MIS前至少含一个有效token的训练序列数	M
algorithm/mis/kept_sequences	Algorithm	段	MIS后至少保留一个有效token的训练序列数	M
algorithm/mis/token_rejection_fraction	Algorithm	0–1	`(original_tokens−kept_tokens)/original_tokens`；本批原始有效token中被过滤的比例	M
algorithm/mis/sequence_rejection_fraction	Algorithm	0–1	`(original_sequences−kept_sequences)/original_sequences`；仅整段无有效token才计整段拒绝	M
algorithm/mis/veto_fraction	Algorithm	0–1	因veto而整段置零的序列数/original_sequences；条件是某个有效token的d低于log(veto阈值)	M
algorithm/mis/clipped_fraction	Algorithm	0–1	被TIS truncate/clip修改权重的token数/original_tokens；TIS mask拒绝不计作clipped	M
algorithm/mis/batch_norm_factor	Algorithm	倍数	TIS后权重实际除以的归一化因子；未归一化或均值过小时为1	M
algorithm/mis/raw_weight_mean	Algorithm	倍数	原始有效token位置上的安全限制后、TIS处理前权重均值，包含随后被拒绝的位置	M
algorithm/mis/raw_weight_min	Algorithm	倍数	上述原始权重最小值	M
algorithm/mis/raw_weight_max	Algorithm	倍数	上述原始权重最大值	M
algorithm/mis/final_weight_mean	Algorithm	倍数	TIS和可选归一化后的权重均值，仍在原始有效token位置统计，包含被拒绝位置	M
algorithm/mis/final_weight_min	Algorithm	倍数	上述最终权重最小值	M
algorithm/mis/final_weight_max	Algorithm	倍数	上述最终权重最大值	M
algorithm/mis/mismatch_tokens	Algorithm	token	概率差异诊断覆盖的原始有效response token数，不是“发现差异的token数”	M
algorithm/mis/mismatch_sequences	Algorithm	段	至少含一个诊断token的训练序列数	M
algorithm/mis/mismatch_groups	Algorithm	组	至少含一个诊断token的group数	M
algorithm/mis/nonfinite_metric_count	Algorithm	指标种类	因某段或聚合值非有限而被省略的指标种类数；不是异常token数，也不是异常段数	M
algorithm/mis/training_log_ppl	Algorithm	nat/token	逐段计算负的trainer重算logprob均值，再按下述模式聚合	M
algorithm/mis/training_ppl	Algorithm	无量纲	逐段先计算exp(负的trainer重算logprob均值)，再聚合；不是最终log-PPL取指数	M
algorithm/mis/rollout_log_ppl	Algorithm	nat/token	逐段计算负的原采样logprob均值，再聚合	M
algorithm/mis/rollout_ppl	Algorithm	无量纲	逐段先计算exp(负的原采样logprob均值)，再聚合	M
algorithm/mis/kl	Algorithm	nat/token	逐段`−mean(d)`再聚合；是rollout与更新前trainer的采样logprob差值估计，可能为负，不是reference KL	M
algorithm/mis/k3_kl	Algorithm	nat/token	逐段`mean(exp(d)−1−d)`再聚合；指数不做安全截断	M
algorithm/mis/log_ppl_abs_diff	Algorithm	nat/token	逐段取`abs(−mean(d))`后再聚合；不是最终algorithm/mis/kl取绝对值	M
algorithm/mis/ppl_ratio	Algorithm	倍数	逐段`exp(−mean(d))`再聚合，即该段trainer PPL/rollout PPL的聚合值；不是两个最终PPL直接相除	M
algorithm/mis/chi2_token	Algorithm	无量纲	逐段`mean(exp(clamp(d,−20,20))²)−1`再聚合；有安全限制的二阶概率比率诊断	M
algorithm/mis/chi2_seq	Algorithm	无量纲	逐段`exp(clamp(sum(d),−20,20))²−1`再聚合；与分段边界有关	M
algorithm/mis/skipped_batches	Algorithm	批	MIS后分出的optimizer batch中，因无有效训练目标而被排除的batch数	C→D
algorithm/mis/optimizer_steps	Algorithm	批	过滤后计划执行的optimizer batch数；当前代码采集时尚未执行，不能标成已成功更新次数	C→D
algorithm/opd/teacher_kl	Algorithm	nat/token	已获得student/teacher logprob且成功对齐的response位置上，sum(log p_student−log p_teacher)/有效token总数；按token加权，是采样reverse-KL估计	OPD prepare，C之前
algorithm/opd/valid_tokens	Algorithm	token	实际参与上述teacher_kl的response token总数；缺失logprob或对齐失败的序列不计入	OPD prepare，C之前
"""
