# 가속 장치 실행

## 생성

Intel NPU와 GPU는 OpenVINO GenAI 형식의 모델 디렉터리를 사용합니다. 프로젝트 루트에서 OpenVINO용 가상환경을 활성화합니다.

```powershell
. .training/openvino-venv/Scripts/Activate.ps1
py generate.py training/inventory-label.schema.json --ov-model .training/qwen3-npu-int4 --device npu --language ko --count 1
```

`--device gpu`는 OpenVINO GPU를 지정합니다. 모델 디렉터리를 함께 지정한 상태에서 `--device auto`를 사용하면 초기화 가능한 장치를 NPU, GPU, CPU 순으로 선택합니다. 명시적으로 지정한 장치의 초기화에 실패하면 오류가 출력됩니다. 실제 선택한 장치는 표준 오류로 표시됩니다. 기존 GGUF 실행은 `--ov-model`을 생략하면 사용할 수 있습니다.

OpenVINO 모델을 준비하려면 OpenVINO GenAI와 Optimum Intel을 설치한 환경에서 아래와 같이 변환합니다. 현재 학습 기반 모델은 `.training/base`에 있습니다.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
optimum-cli export openvino -m .training/base --task text-generation-with-past --weight-format int4 --sym --ratio 1.0 --group-size 128 .training/qwen3-npu-int4
```

위 명령은 기본 모델을 변환합니다. 학습 결과까지 OpenVINO에서 쓰려면 Hugging Face 원본에 PEFT 어댑터를 병합한 모델을 변환 입력으로 사용합니다. 기존 GGUF 어댑터는 기존 GGUF 경로에서 `--adapter`로 적용할 수 있습니다. [Intel NPU 변환 문서](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)

Intel AI Boost에서 장치 연산과 위 생성 명령을 실행했습니다. 실행기가 `NPU`를 표시했고 [샘플 JSON](../outputs/npu-smoke.json)이 스키마 검증을 통과했습니다. 기본 모델이 생성한 주방용품명은 `errors:123`으로 내용 조건을 충족하지 못했습니다.

Intel Arc에서도 같은 명령에 `--device gpu`를 지정해 실행했습니다. 장치 표시가 `GPU`였고 [출력 JSON](../outputs/gpu-smoke.json)이 검증을 통과했습니다. 값은 NPU 샘플과 같았습니다.

## GGUF 모델 재사용

Vulkan 배포본의 `llama-server.exe`로 명령 한 번 동안 GGUF와 LoRA를 GPU에 올려 두고 재사용할 수 있습니다. 현재 로컬에 설치한 Vulkan 실행기와 EXAONE을 쓰는 예입니다.

```powershell
py generate.py training/source/context-event.schema.json --server-runner .training/llama-vulkan/llama-server.exe --device gpu --model .training/exaone-4.0-1.2b-q4.gguf --adapter .training/exaone-scenario-lora.gguf --language ko --count 5 --scenario "초보자 대상 도자기 체험 행사입니다."
```

실행 중에만 로컬 작업 프로세스가 열립니다. 시작 시간과 GPU에 배치한 레이어 수는 표준 오류에 표시됩니다. 현재는 필드 요청을 순서대로 처리하고 같은 입력 앞부분을 재사용합니다. GPU를 지정했는데 레이어 배치가 확인되지 않으면 오류를 반환합니다.

## 학습

`training/train.py --device auto`는 현재 PyTorch 환경에서 사용 가능한 CUDA, Intel XPU, CPU를 순서대로 선택합니다. `--device cuda`, `--device xpu`, `--device cpu`로 지정할 수도 있습니다. Intel Arc 학습에는 XPU 지원 PyTorch가 필요합니다. 이 프로젝트에서 Intel NPU는 OpenVINO 추론 경로로 사용합니다. [PyTorch XPU 설치 및 확인](https://docs.pytorch.org/get-started/additional-platforms/)

이 노트북의 `.training/xpu-venv`에서 Intel Arc 역전파, LoRA 1스텝, 평가 26건과 어댑터 저장을 확인했습니다. 재실행할 때는 새 출력 경로를 지정합니다.

```powershell
. .training/xpu-venv/Scripts/Activate.ps1
py training/train.py --device xpu --steps 1 --out .training/xpu-smoke-v3
```
