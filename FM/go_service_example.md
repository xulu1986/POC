# Go Service Example for FM Model Inference

This document shows how to use the JIT-compiled FM model in a Go service.

## Model Format

The model is saved as a JSON file with this structure:
```json
{
  "model": "UEsDBBQAAAAIAG...",  // Base64-encoded JIT scripted model
  "config": "{\"num_numerical_features\": 5, \"embedding_dim\": 8}"  // JSON string
}
```

## Go Service Implementation

### 1. Dependencies

```go
// go.mod
module fm-service

go 1.19

require (
    github.com/pytorch/pytorch/torch/csrc/api v1.13.0  // PyTorch C++ API
    github.com/gin-gonic/gin v1.9.1                    // Web framework
)
```

### 2. Model Loading

```go
package main

import (
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "log"
    
    "github.com/pytorch/pytorch/torch/csrc/api"  // PyTorch Go bindings
)

type ModelData struct {
    Model  string `json:"model"`   // Base64-encoded JIT model
    Config string `json:"config"`  // JSON string config
}

type ModelConfig struct {
    NumNumericalFeatures   int   `json:"num_numerical_features"`
    NumCategoricalFeatures int   `json:"num_categorical_features"`
    CategoricalVocabSizes  []int `json:"categorical_vocab_sizes"`
    EmbeddingDim          int   `json:"embedding_dim"`
    TaskType              string `json:"task_type"`
}

type FMService struct {
    model  torch.ScriptModule
    config ModelConfig
}

func NewFMService(modelPath string) (*FMService, error) {
    // Read model file
    data, err := ioutil.ReadFile(modelPath)
    if err != nil {
        return nil, fmt.Errorf("failed to read model file: %v", err)
    }
    
    // Parse JSON
    var modelData ModelData
    if err := json.Unmarshal(data, &modelData); err != nil {
        return nil, fmt.Errorf("failed to parse model JSON: %v", err)
    }
    
    // Decode base64 model
    modelBytes, err := base64.StdEncoding.DecodeString(modelData.Model)
    if err != nil {
        return nil, fmt.Errorf("failed to decode model: %v", err)
    }
    
    // Load JIT scripted model
    model, err := torch.LoadScriptModule(modelBytes)
    if err != nil {
        return nil, fmt.Errorf("failed to load JIT scripted model: %v", err)
    }
    
    // Parse config
    var config ModelConfig
    if err := json.Unmarshal([]byte(modelData.Config), &config); err != nil {
        return nil, fmt.Errorf("failed to parse config: %v", err)
    }
    
    return &FMService{
        model:  model,
        config: config,
    }, nil
}
```

### 3. Prediction Methods

```go
type PredictionRequest struct {
    NumericalFeatures   [][]float32 `json:"numerical_features"`   // Shape: [batch_size, num_numerical]
    CategoricalFeatures [][]int64   `json:"categorical_features"` // Shape: [batch_size, num_categorical]
}

type PredictionResponse struct {
    Predictions []float32 `json:"predictions"`
    BatchSize   int       `json:"batch_size"`
}

func (s *FMService) Predict(req PredictionRequest) (*PredictionResponse, error) {
    batchSize := len(req.NumericalFeatures)
    if batchSize == 0 {
        return nil, fmt.Errorf("empty batch")
    }
    
    // Validate input dimensions
    if len(req.NumericalFeatures[0]) != s.config.NumNumericalFeatures {
        return nil, fmt.Errorf("expected %d numerical features, got %d", 
            s.config.NumNumericalFeatures, len(req.NumericalFeatures[0]))
    }
    
    if len(req.CategoricalFeatures[0]) != s.config.NumCategoricalFeatures {
        return nil, fmt.Errorf("expected %d categorical features, got %d", 
            s.config.NumCategoricalFeatures, len(req.CategoricalFeatures[0]))
    }
    
    // Convert to tensors
    numericalTensor := torch.FromBlob(
        unsafe.Pointer(&req.NumericalFeatures[0][0]),
        []int64{int64(batchSize), int64(s.config.NumNumericalFeatures)},
        torch.Float32,
    )
    
    categoricalTensor := torch.FromBlob(
        unsafe.Pointer(&req.CategoricalFeatures[0][0]),
        []int64{int64(batchSize), int64(s.config.NumCategoricalFeatures)},
        torch.Int64,
    )
    
    // Run inference
    inputs := []torch.Tensor{numericalTensor, categoricalTensor}
    outputs, err := s.model.Forward(inputs)
    if err != nil {
        return nil, fmt.Errorf("inference failed: %v", err)
    }
    
    // Extract predictions
    predictions := make([]float32, batchSize)
    outputData := outputs[0].Data()
    for i := 0; i < batchSize; i++ {
        predictions[i] = outputData[i]
    }
    
    return &PredictionResponse{
        Predictions: predictions,
        BatchSize:   batchSize,
    }, nil
}

// Single prediction convenience method
func (s *FMService) PredictSingle(numerical []float32, categorical []int64) (float32, error) {
    req := PredictionRequest{
        NumericalFeatures:   [][]float32{numerical},
        CategoricalFeatures: [][]int64{categorical},
    }
    
    resp, err := s.Predict(req)
    if err != nil {
        return 0, err
    }
    
    return resp.Predictions[0], nil
}
```

### 4. HTTP API

```go
import (
    "net/http"
    "github.com/gin-gonic/gin"
)

func (s *FMService) SetupRoutes() *gin.Engine {
    r := gin.Default()
    
    // Health check
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
            "config": s.config,
        })
    })
    
    // Batch prediction
    r.POST("/predict", func(c *gin.Context) {
        var req PredictionRequest
        if err := c.ShouldBindJSON(&req); err != nil {
            c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
            return
        }
        
        resp, err := s.Predict(req)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
            return
        }
        
        c.JSON(http.StatusOK, resp)
    })
    
    // Single prediction
    r.POST("/predict/single", func(c *gin.Context) {
        var req struct {
            Numerical   []float32 `json:"numerical"`
            Categorical []int64   `json:"categorical"`
        }
        
        if err := c.ShouldBindJSON(&req); err != nil {
            c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
            return
        }
        
        prediction, err := s.PredictSingle(req.Numerical, req.Categorical)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
            return
        }
        
        c.JSON(http.StatusOK, gin.H{
            "prediction": prediction,
        })
    })
    
    return r
}

func main() {
    // Load model
    service, err := NewFMService("production_model.json")
    if err != nil {
        log.Fatal("Failed to load model:", err)
    }
    
    // Setup routes
    r := service.SetupRoutes()
    
    // Start server
    log.Println("Starting FM service on :8080")
    r.Run(":8080")
}
```

## Usage Examples

### Single Prediction
```bash
curl -X POST http://localhost:8080/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "numerical": [1.5, 2.3, 0.8, 1.1, 0.5],
    "categorical": [2, 15, 7]
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "numerical_features": [
      [1.5, 2.3, 0.8, 1.1, 0.5],
      [0.8, 1.2, 1.5, 0.9, 1.3],
      [2.1, 0.7, 1.8, 1.4, 0.6]
    ],
    "categorical_features": [
      [2, 15, 7],
      [1, 8, 12],
      [3, 19, 5]
    ]
  }'
```

## Key Benefits

1. **Flexible Batch Sizes**: The JIT scripted model supports any batch size (1, 4, 16, 64, etc.)
2. **High Performance**: JIT scripting optimizes the model for inference
3. **Self-contained**: Model file includes both weights and configuration
4. **Type Safety**: Go provides compile-time type checking
5. **Easy Deployment**: Single JSON file contains everything needed

## Performance Tips

1. **Batch Processing**: Process multiple requests together when possible
2. **Connection Pooling**: Reuse the model instance across requests
3. **Memory Management**: Use tensor views to avoid copying data
4. **Async Processing**: Handle multiple requests concurrently

The JIT scripting approach ensures your Go service can handle any batch size efficiently! 