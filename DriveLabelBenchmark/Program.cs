using RapidOCRLib;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;

Console.OutputEncoding = Encoding.UTF8;

Console.WriteLine("==============================================");
Console.WriteLine(" DEP Drive Label OCR Benchmark");
Console.WriteLine(" RapidOCR / PP-OCRv5");
Console.WriteLine("==============================================");
Console.WriteLine();

if (args.Length == 0)
{
    Console.WriteLine("Usage:");
    Console.WriteLine("  DriveLabelBenchmark.exe <image-or-folder>");
    Console.WriteLine();
    Console.WriteLine("Examples:");
    Console.WriteLine(@"  DriveLabelBenchmark.exe C:\OCR-Test\IMG_001.jpg");
    Console.WriteLine(@"  DriveLabelBenchmark.exe C:\OCR-Test");
    return;
}

string inputPath = Path.GetFullPath(args[0]);

if (!File.Exists(inputPath) && !Directory.Exists(inputPath))
{
    Console.WriteLine($"ERROR: Input not found:");
    Console.WriteLine(inputPath);
    return;
}

// ----------------------------------------------------
// Locate OCR models
// ----------------------------------------------------

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
    string fullPath = Path.Combine(modelPath, model);

    if (!File.Exists(fullPath))
    {
        Console.WriteLine($"ERROR: Missing OCR model:");
        Console.WriteLine(fullPath);
        return;
    }
}

// ----------------------------------------------------
// Initialize RapidOCR ONCE
// ----------------------------------------------------

Console.WriteLine("Initializing RapidOCR models...");

var initWatch = Stopwatch.StartNew();

OcrLite ocrEngine = new OcrLite()
{
    DetPath = Path.Combine(modelPath, "ch_PP-OCRv5_mobile_det.onnx"),
    ClsPath = Path.Combine(modelPath, "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
    RecPath = Path.Combine(modelPath, "ch_PP-OCRv5_rec_mobile_infer.onnx"),
    KeyDicPath = Path.Combine(modelPath, "ppocrv5_dict.txt"),
};

await ocrEngine.InitModels();

initWatch.Stop();

Console.WriteLine(
    $"Models initialized in {initWatch.Elapsed.TotalSeconds:F2} sec.");
Console.WriteLine();

// ----------------------------------------------------
// Build image list
// ----------------------------------------------------

HashSet<string> supportedExtensions =
    new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    };

List<string> images = new();

if (File.Exists(inputPath))
{
    if (!supportedExtensions.Contains(Path.GetExtension(inputPath)))
    {
        Console.WriteLine(
            $"ERROR: Unsupported image type: {Path.GetExtension(inputPath)}");
        return;
    }

    images.Add(inputPath);
}
else
{
    images = Directory
        .EnumerateFiles(inputPath)
        .Where(f => supportedExtensions.Contains(Path.GetExtension(f)))
        .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
        .ToList();
}

if (images.Count == 0)
{
    Console.WriteLine("No supported image files found.");
    return;
}

Console.WriteLine($"Images found: {images.Count}");
Console.WriteLine();

// ----------------------------------------------------
// Output directory
// ----------------------------------------------------

string sourceDirectory =
    File.Exists(inputPath)
        ? Path.GetDirectoryName(inputPath)!
        : inputPath;

string timestamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");

string outputDirectory =
    Path.Combine(
        sourceDirectory,
        $"RapidOCR_Benchmark_{timestamp}");

Directory.CreateDirectory(outputDirectory);

string csvFile =
    Path.Combine(outputDirectory, "RapidOCR_Detections.csv");

string imageCsvFile =
    Path.Combine(outputDirectory, "RapidOCR_Images.csv");

string jsonFile =
    Path.Combine(outputDirectory, "RapidOCR_Results.json");

string textDirectory =
    Path.Combine(outputDirectory, "RawText");

Directory.CreateDirectory(textDirectory);

// ----------------------------------------------------
// Result objects
// ----------------------------------------------------

List<ImageResult> allResults = new();

Stopwatch totalWatch = Stopwatch.StartNew();

int imageNumber = 0;

// ----------------------------------------------------
// Process images
// ----------------------------------------------------

foreach (string image in images)
{
    imageNumber++;

    Console.WriteLine(
        $"[{imageNumber}/{images.Count}] {Path.GetFileName(image)}");

    Stopwatch imageWatch = Stopwatch.StartNew();

    try
    {
        var result = ocrEngine.Detect(
    image,
    padding: 50,
    maxSideLen: 1024,
    boxScoreThresh: 0.5f,
    boxThresh: 0.3f,
    unClipRatio: 1.6f,
    doAngle: false,
    mostAngle: false
);

        imageWatch.Stop();

        ImageResult imageResult = new()
        {
            FileName = Path.GetFileName(image),
            FullPath = image,
            ElapsedMilliseconds = imageWatch.Elapsed.TotalMilliseconds,
            RapidDetectMilliseconds = result?.DetectTime ?? 0,
            RawText = result?.StrRes?.Trim() ?? ""
        };

        if (result?.TextBlocks != null)
        {
            int blockNumber = 0;

            foreach (var block in result.TextBlocks)
            {
                blockNumber++;

                double averageCharConfidence = 0;

                if (block.CharScores != null &&
                    block.CharScores.Count > 0)
                {
                    averageCharConfidence =
                        block.CharScores.Average();
                }

                string coordinates = "";

                if (block.BoxPoints != null)
                {
                    coordinates = string.Join(
                        " | ",
                        block.BoxPoints.Select(
                            p => $"{p.X},{p.Y}"));
                }

                imageResult.Detections.Add(
                    new DetectionResult
                    {
                        Block = blockNumber,
                        Text = block.Text ?? "",
                        BoxConfidence = block.BoxScore,
                        AverageCharacterConfidence =
                            averageCharConfidence,
                        Coordinates = coordinates,
                        RecognitionMilliseconds =
                            block.CrnnTime,
                        BlockMilliseconds =
                            block.BlockTime
                    });
            }
        }

        allResults.Add(imageResult);

        string rawTextFile =
            Path.Combine(
                textDirectory,
                Path.GetFileNameWithoutExtension(image)
                + ".txt");

        await File.WriteAllTextAsync(
            rawTextFile,
            imageResult.RawText,
            Encoding.UTF8);

        Console.WriteLine(
            $"    {imageResult.Detections.Count} blocks | " +
            $"{imageResult.ElapsedMilliseconds:F0} ms | " +
            $"{imageResult.RawText.Length} chars");
    }
    catch (Exception ex)
    {
        imageWatch.Stop();

        allResults.Add(
            new ImageResult
            {
                FileName = Path.GetFileName(image),
                FullPath = image,
                ElapsedMilliseconds =
                    imageWatch.Elapsed.TotalMilliseconds,
                Error = ex.ToString()
            });

        Console.WriteLine($"    ERROR: {ex.Message}");
    }
}

totalWatch.Stop();

// ----------------------------------------------------
// Write per-detection CSV
// ----------------------------------------------------

StringBuilder detectionCsv = new();

detectionCsv.AppendLine(
    "FileName,Block,Text,BoxConfidence," +
    "AverageCharacterConfidence,Coordinates," +
    "RecognitionMilliseconds,BlockMilliseconds," +
    "ImageElapsedMilliseconds,RapidDetectMilliseconds");

foreach (ImageResult image in allResults)
{
    foreach (DetectionResult detection in image.Detections)
    {
        detectionCsv.AppendLine(
            string.Join(",",
                Csv(image.FileName),
                detection.Block,
                Csv(detection.Text),
                F(detection.BoxConfidence),
                F(detection.AverageCharacterConfidence),
                Csv(detection.Coordinates),
                F(detection.RecognitionMilliseconds),
                F(detection.BlockMilliseconds),
                F(image.ElapsedMilliseconds),
                F(image.RapidDetectMilliseconds)
            ));
    }

    if (image.Detections.Count == 0)
    {
        detectionCsv.AppendLine(
            string.Join(",",
                Csv(image.FileName),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                F(image.ElapsedMilliseconds),
                F(image.RapidDetectMilliseconds)
            ));
    }
}

await File.WriteAllTextAsync(
    csvFile,
    detectionCsv.ToString(),
    Encoding.UTF8);

// ----------------------------------------------------
// Write per-image CSV
// ----------------------------------------------------

StringBuilder imageCsv = new();

imageCsv.AppendLine(
    "FileName,DetectionCount,ElapsedMilliseconds," +
    "RapidDetectMilliseconds,RawText,Error");

foreach (ImageResult image in allResults)
{
    imageCsv.AppendLine(
        string.Join(",",
            Csv(image.FileName),
            image.Detections.Count,
            F(image.ElapsedMilliseconds),
            F(image.RapidDetectMilliseconds),
            Csv(image.RawText),
            Csv(image.Error)
        ));
}

await File.WriteAllTextAsync(
    imageCsvFile,
    imageCsv.ToString(),
    Encoding.UTF8);

// ----------------------------------------------------
// Write JSON
// ----------------------------------------------------

JsonSerializerOptions jsonOptions = new()
{
    WriteIndented = true
};

await File.WriteAllTextAsync(
    jsonFile,
    JsonSerializer.Serialize(allResults, jsonOptions),
    Encoding.UTF8);

// ----------------------------------------------------
// Summary
// ----------------------------------------------------

int succeeded =
    allResults.Count(r => string.IsNullOrEmpty(r.Error));

int failed =
    allResults.Count - succeeded;

double averageMs =
    succeeded > 0
        ? allResults
            .Where(r => string.IsNullOrEmpty(r.Error))
            .Average(r => r.ElapsedMilliseconds)
        : 0;

Console.WriteLine();
Console.WriteLine("==============================================");
Console.WriteLine(" BENCHMARK COMPLETE");
Console.WriteLine("==============================================");
Console.WriteLine($"Images:          {allResults.Count}");
Console.WriteLine($"Succeeded:       {succeeded}");
Console.WriteLine($"Failed:          {failed}");
Console.WriteLine($"Average OCR:     {averageMs:F0} ms/image");
Console.WriteLine(
    $"Total OCR time:  {totalWatch.Elapsed.TotalSeconds:F2} sec");
Console.WriteLine();
Console.WriteLine("Results:");
Console.WriteLine(outputDirectory);
Console.WriteLine();
Console.WriteLine("Files created:");
Console.WriteLine("  RapidOCR_Images.csv");
Console.WriteLine("  RapidOCR_Detections.csv");
Console.WriteLine("  RapidOCR_Results.json");
Console.WriteLine("  RawText\\");
Console.WriteLine();

// ----------------------------------------------------
// Helpers / models
// ----------------------------------------------------

static string Csv(string? value)
{
    if (string.IsNullOrEmpty(value))
        return "\"\"";

    return "\"" +
        value.Replace("\"", "\"\"")
        + "\"";
}

static string F(double value)
{
    return value.ToString(
        "0.###",
        CultureInfo.InvariantCulture);
}

class ImageResult
{
    public string FileName { get; set; } = "";
    public string FullPath { get; set; } = "";
    public double ElapsedMilliseconds { get; set; }
    public double RapidDetectMilliseconds { get; set; }
    public string RawText { get; set; } = "";
    public string Error { get; set; } = "";
    public List<DetectionResult> Detections { get; set; } = new();
}

class DetectionResult
{
    public int Block { get; set; }
    public string Text { get; set; } = "";
    public double BoxConfidence { get; set; }
    public double AverageCharacterConfidence { get; set; }
    public string Coordinates { get; set; } = "";
    public double RecognitionMilliseconds { get; set; }
    public double BlockMilliseconds { get; set; }
}
