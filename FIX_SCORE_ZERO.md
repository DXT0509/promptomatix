# 🐛 Fix: Score = 0.0000 Issue

## Vấn Đề

Khi chạy prompt optimization, bạn thấy:
```
📊 Performance Metrics
Initial Score: 0.0000
Optimized Score: 0.0000
```

## Nguyên Nhân

Vấn đề xảy ra khi **`input_fields` và `output_fields` không được set**, dẫn đến:

1. **Metrics không biết field nào cần đánh giá**
   - Code: `MetricsManager._get_output_value()` trả về chuỗi rỗng
   - Kết quả: Không thể so sánh prediction vs ground truth

2. **Evaluation bị lỗi âm thầm**
   - Mỗi sample evaluation throw exception
   - Exception bị `continue` bỏ qua
   - `valid_evaluations = 0` → return `0.0`

## Ví Dụ Session Bị Lỗi

```json
{
  "config": {
    "input_fields": [],      // ❌ RỖNG!
    "output_fields": [],     // ❌ RỖNG!
    "valid_data": [
      {
        "question": "...",   // ✅ Data có fields
        "answer": "..."      // ✅ Nhưng config không biết
      }
    ]
  }
}
```

## Giải Pháp Đã Implement

### 1. Auto-detect Fields từ Synthetic Data

**File**: `src/promptomatix/core/optimizer.py`

**Changes**:
```python
def _run_meta_prompt_backend(self, initial_flag: bool = True) -> Dict:
    # ... existing code ...
    
    # 🆕 Auto-detect input/output fields if not set
    if (not self.config.input_fields or not self.config.output_fields):
        all_data = self.config.train_data + (self.config.valid_data or [])
        if all_data:
            self._auto_detect_fields(all_data)
            print(f"📋 Auto-detected fields - Input: {self.config.input_fields}, Output: {self.config.output_fields}")
```

**Method mới**:
```python
def _auto_detect_fields(self, synthetic_data: List[Dict]) -> None:
    """Auto-detect input/output fields bằng heuristics"""
    
    # Patterns thường gặp
    output_patterns = ['answer', 'output', 'label', 'response', 'result', 'summary']
    input_patterns = ['question', 'input', 'text', 'query', 'context', 'prompt']
    
    # Match exact patterns
    for key in sample_keys:
        if any(pattern in key.lower() for pattern in output_patterns):
            detected_output.append(key)
        elif any(pattern in key.lower() for pattern in input_patterns):
            detected_input.append(key)
    
    # Fallback: last field = output, others = input
    if not detected_output:
        detected_output = [sample_keys[-1]]
    if not detected_input:
        detected_input = [k for k in sample_keys if k not in detected_output]
```

### 2. Configure MetricsManager Trước Evaluation

```python
def _evaluate_prompt_meta_backend(self, prompt: str) -> float:
    # 🆕 Ensure MetricsManager có output fields
    if self.config.output_fields:
        from ..metrics.metrics import MetricsManager
        MetricsManager.configure(self.config.output_fields)
    
    # ... evaluation code ...
```

## Cách Test Fix

### Option 1: Chạy Script Test

```bash
python test_fix.py
```

Script sẽ:
- Tạo một optimization task đơn giản
- Kiểm tra xem scores có > 0 không
- In ra debug info nếu vẫn fail

### Option 2: Manual Test

```python
from promptomatix.main import process_input

result = process_input(
    raw_input="Answer questions briefly",
    synthetic_data_size=3,
    task_type="generation",
    backend="simple_meta_prompt"
)

print(f"Initial: {result['metrics']['initial_prompt_score']}")
print(f"Optimized: {result['metrics']['optimized_prompt_score']}")
```

### Expected Output

```
📋 Auto-detected fields - Input: ['question'], Output: ['answer']
🔧 Evaluating initial prompt...
  Initial score: 0.6234
📊 Evaluating optimized prompt...
  Optimized score: 0.7891
✅ Prompt optimization complete!
```

## Kiểm Tra Session Cũ

Để xem session cũ có vấn đề gì:

```python
import json

with open('sessions/YOUR_SESSION_ID.json') as f:
    session = json.load(f)
    
config = session['config']
print(f"Input fields: {config.get('input_fields', [])}")
print(f"Output fields: {config.get('output_fields', [])}")

if config.get('valid_data'):
    sample = config['valid_data'][0]
    print(f"Sample keys: {list(sample.keys())}")
```

## Debug Tips

Nếu vẫn bị `0.0000`:

### 1. Check Synthetic Data

```python
# In ra synthetic data để xem structure
if result.get('synthetic_data'):
    print(json.dumps(result['synthetic_data'][0], indent=2))
```

### 2. Check Fields Detection

```python
# Add logging vào _auto_detect_fields
print(f"Sample keys: {sample_keys}")
print(f"Detected input: {detected_input}")
print(f"Detected output: {detected_output}")
```

### 3. Check Evaluation

Thêm debug vào `_evaluate_prompt_meta_backend`:

```python
for i, sample in enumerate(all_data):
    try:
        # ... evaluation ...
        print(f"Sample {i}: score = {score}")
    except Exception as e:
        print(f"Sample {i} FAILED: {str(e)}")  # 🆕 In lỗi thay vì bỏ qua
        continue
```

## Lưu Ý

1. **Auto-detection chỉ áp dụng cho `simple_meta_prompt` backend**
   - DSPy backend có logic riêng để extract fields

2. **Heuristics có thể fail với tên field không chuẩn**
   - Ví dụ: `my_custom_answer` có thể không được detect
   - Trong trường hợp này, user nên pass `input_fields` và `output_fields` explicitly

3. **Fields phải match với synthetic data**
   - Nếu synthetic data có `["query", "response"]`
   - Nhưng config detect `["question", "answer"]`
   - → Evaluation sẽ fail

## Cải Tiến Trong Tương Lai

- [ ] Add validation để check fields có tồn tại trong data không
- [ ] Throw warning nếu auto-detection fallback to default
- [ ] Support custom field mapping
- [ ] Better error messages khi evaluation fails
- [ ] Retry logic với different field combinations

## Tài Liệu Liên Quan

- [metrics.py](src/promptomatix/metrics/metrics.py) - Metrics implementation
- [optimizer.py](src/promptomatix/core/optimizer.py) - Optimization logic
- [config.py](src/promptomatix/core/config.py) - Fields extraction

---

## ✅ Test Results

**Status**: **FIXED** ✅

```bash
$ python test_fix.py

Initial Score: 0.8632  ✅ (was 0.0000)
Optimized Score: 0.7950 ✅ (was 0.0000)
✅ PASS: Scores are non-zero!
```

**Verification**:
- ✅ Fields auto-detected: `input_fields: ['question'], output_fields: ['answer']`
- ✅ MetricsManager configured correctly
- ✅ BERTScore working: fluency=0.84, creativity=0.84, similarity=0.87
- ✅ All samples evaluated successfully (3/3 valid)

---

**Last Updated**: 2025-10-22  
**Author**: GitHub Copilot  
**Status**: ✅ Fixed & Tested
