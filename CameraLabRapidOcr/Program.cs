using RapidOCRLib;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;

Console.OutputEncoding = Encoding.UTF8;

string modelPath = Path.Combine(AppContext.BaseDirectory, "models");
string[] requiredModels =
{
    "ch_PP-OCRv5_mobile_det.onnx",
    "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "ch_PP-OCRv5_rec_mobile_infer.onnx",
    "ppocrv5_dict.txt"
};

foreach (string model in requiredModels)
{
    string full = Path.Combine(modelPath, model);
    if (!File.Exists(full))
    {
        Emit(new BridgeResponse { Ok = false, Error = $"Missing OCR model: {full}" });
        return;
    }
}

var initWatch = Stopwatch.StartNew();
var ocr = new OcrLite
{
    DetPath = Path.Combine(modelPath, requiredModels[0]),
    ClsPath = Path.Combine(modelPath, requiredModels[1]),
    RecPath = Path.Combine(modelPath, requiredModels[2]),
    KeyDicPath = Path.Combine(modelPath, requiredModels[3])
};
await ocr.InitModels();
initWatch.Stop();

// One JSON request per line in, one JSON response per line out.
// Keeping this process alive means Camera Lab pays model startup only once.
string? requestLine;
while ((requestLine = Console.ReadLine()) != null)
{
    if (string.IsNullOrWhiteSpace(requestLine)) continue;
    BridgeRequest? request = null;
    try
    {
        request = JsonSerializer.Deserialize<BridgeRequest>(requestLine,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        if (request == null || string.IsNullOrWhiteSpace(request.Path))
            throw new InvalidOperationException("Request must include Path.");
        if (!File.Exists(request.Path))
            throw new FileNotFoundException("Image not found.", request.Path);

        var sw = Stopwatch.StartNew();
        var result = ocr.Detect(
            request.Path,
            padding: request.Padding ?? 50,
            maxSideLen: request.MaxSideLen ?? 1024,
            boxScoreThresh: request.BoxScoreThresh ?? 0.5f,
            boxThresh: request.BoxThresh ?? 0.3f,
            unClipRatio: request.UnClipRatio ?? 1.6f,
            doAngle: false,
            mostAngle: false);
        sw.Stop();

        var response = new BridgeResponse
        {
            Ok = true,
            Path = request.Path,
            Engine = "RapidOCR/PP-OCRv5",
            InitMilliseconds = initWatch.Elapsed.TotalMilliseconds,
            ElapsedMilliseconds = sw.Elapsed.TotalMilliseconds,
            RapidDetectMilliseconds = result?.DetectTime ?? 0,
            Text = result?.StrRes?.Trim() ?? ""
        };

        if (result?.TextBlocks != null)
        {
            int index = 0;
            foreach (var block in result.TextBlocks)
            {
                index++;
                var points = block.BoxPoints?.Select(p => new BridgePoint { X = p.X, Y = p.Y }).ToList()
                             ?? new List<BridgePoint>();
                float avgChar = block.CharScores != null && block.CharScores.Count > 0
                    ? block.CharScores.Average()
                    : 0;
                response.Blocks.Add(new BridgeBlock
                {
                    Index = index,
                    Text = block.Text ?? "",
                    BoxConfidence = block.BoxScore,
                    AverageCharacterConfidence = avgChar,
                    RecognitionMilliseconds = block.CrnnTime,
                    BlockMilliseconds = block.BlockTime,
                    Points = points
                });
            }
        }
        Emit(response);
    }
    catch (Exception ex)
    {
        Emit(new BridgeResponse
        {
            Ok = false,
            Path = request?.Path ?? "",
            Engine = "RapidOCR/PP-OCRv5",
            InitMilliseconds = initWatch.Elapsed.TotalMilliseconds,
            Error = ex.ToString()
        });
    }
}

static void Emit(BridgeResponse response)
{
    Console.WriteLine(JsonSerializer.Serialize(response));
    Console.Out.Flush();
}

class BridgeRequest
{
    public string Path { get; set; } = "";
    public int? Padding { get; set; }
    public int? MaxSideLen { get; set; }
    public float? BoxScoreThresh { get; set; }
    public float? BoxThresh { get; set; }
    public float? UnClipRatio { get; set; }
}

class BridgeResponse
{
    public bool Ok { get; set; }
    public string Path { get; set; } = "";
    public string Engine { get; set; } = "";
    public double InitMilliseconds { get; set; }
    public double ElapsedMilliseconds { get; set; }
    public double RapidDetectMilliseconds { get; set; }
    public string Text { get; set; } = "";
    public string Error { get; set; } = "";
    public List<BridgeBlock> Blocks { get; set; } = new();
}

class BridgeBlock
{
    public int Index { get; set; }
    public string Text { get; set; } = "";
    public float BoxConfidence { get; set; }
    public float AverageCharacterConfidence { get; set; }
    public double RecognitionMilliseconds { get; set; }
    public double BlockMilliseconds { get; set; }
    public List<BridgePoint> Points { get; set; } = new();
}

class BridgePoint
{
    public float X { get; set; }
    public float Y { get; set; }
}
