using RapidOCRLib;
using System.Diagnostics;
using System.Globalization;
using System.Text;

Console.OutputEncoding = Encoding.UTF8;

if (args.Length == 0)
{
    Console.WriteLine("Usage: DriveLabelSweep.exe <image-or-folder> [maxSideLens]");
    Console.WriteLine("Example: DriveLabelSweep.exe C:\\OCR-Test 1024,1600,2048");
    return;
}

string inputPath = Path.GetFullPath(args[0]);
int[] maxSideLens = args.Length > 1
    ? args[1].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .Select(int.Parse).Distinct().OrderBy(x => x).ToArray()
    : new[] { 1024, 1600, 2048 };

if (!File.Exists(inputPath) && !Directory.Exists(inputPath))
{
    Console.WriteLine($"ERROR: Input not found: {inputPath}");
    return;
}

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
        Console.WriteLine($"ERROR: Missing OCR model: {full}");
        return;
    }
}

HashSet<string> supported = new(StringComparer.OrdinalIgnoreCase)
{ ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff" };

List<string> images = File.Exists(inputPath)
    ? new List<string> { inputPath }
    : Directory.EnumerateFiles(inputPath)
        .Where(f => supported.Contains(Path.GetExtension(f)))
        .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
        .ToList();

if (images.Count == 0)
{
    Console.WriteLine("No supported image files found.");
    return;
}

Console.WriteLine("==============================================");
Console.WriteLine(" DEP Drive Label OCR Resolution Sweep");
Console.WriteLine(" RapidOCR / PP-OCRv5");
Console.WriteLine("==============================================");
Console.WriteLine($"Images: {images.Count}");
Console.WriteLine($"maxSideLen values: {string.Join(", ", maxSideLens)}");
Console.WriteLine();

var initWatch = Stopwatch.StartNew();
OcrLite ocrEngine = new()
{
    DetPath = Path.Combine(modelPath, "ch_PP-OCRv5_mobile_det.onnx"),
    ClsPath = Path.Combine(modelPath, "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
    RecPath = Path.Combine(modelPath, "ch_PP-OCRv5_rec_mobile_infer.onnx"),
    KeyDicPath = Path.Combine(modelPath, "ppocrv5_dict.txt"),
};
await ocrEngine.InitModels();
initWatch.Stop();
Console.WriteLine($"Models initialized in {initWatch.Elapsed.TotalSeconds:F2} sec.");
Console.WriteLine();

string sourceDirectory = File.Exists(inputPath) ? Path.GetDirectoryName(inputPath)! : inputPath;
string rootOutput = Path.Combine(sourceDirectory, $"RapidOCR_Sweep_{DateTime.Now:yyyyMMdd-HHmmss}");
Directory.CreateDirectory(rootOutput);

List<SweepRow> rows = new();

foreach (int maxSideLen in maxSideLens)
{
    Console.WriteLine($"--- maxSideLen={maxSideLen} ---");
    string settingDir = Path.Combine(rootOutput, $"maxSideLen_{maxSideLen}");
    string rawDir = Path.Combine(settingDir, "RawText");
    Directory.CreateDirectory(rawDir);

    int n = 0;
    foreach (string image in images)
    {
        n++;
        var sw = Stopwatch.StartNew();
        try
        {
            var result = ocrEngine.Detect(
                image,
                padding: 50,
                maxSideLen: maxSideLen,
                boxScoreThresh: 0.5f,
                boxThresh: 0.3f,
                unClipRatio: 1.6f,
                doAngle: false,
                mostAngle: false);
            sw.Stop();

            string raw = result?.StrRes?.Trim() ?? "";
            int blocks = result?.TextBlocks?.Count ?? 0;
            double rapidMs = result?.DetectTime ?? 0;

            rows.Add(new SweepRow
            {
                MaxSideLen = maxSideLen,
                FileName = Path.GetFileName(image),
                ElapsedMilliseconds = sw.Elapsed.TotalMilliseconds,
                RapidDetectMilliseconds = rapidMs,
                DetectionCount = blocks,
                RawText = raw
            });

            await File.WriteAllTextAsync(
                Path.Combine(rawDir, Path.GetFileNameWithoutExtension(image) + ".txt"),
                raw,
                Encoding.UTF8);

            Console.WriteLine($"[{n}/{images.Count}] {Path.GetFileName(image)} | {sw.Elapsed.TotalMilliseconds:F0} ms | {blocks} blocks");
        }
        catch (Exception ex)
        {
            sw.Stop();
            rows.Add(new SweepRow
            {
                MaxSideLen = maxSideLen,
                FileName = Path.GetFileName(image),
                ElapsedMilliseconds = sw.Elapsed.TotalMilliseconds,
                Error = ex.ToString()
            });
            Console.WriteLine($"[{n}/{images.Count}] {Path.GetFileName(image)} | ERROR: {ex.Message}");
        }
    }
    Console.WriteLine();
}

string resultsCsv = Path.Combine(rootOutput, "SweepResults.csv");
var csv = new StringBuilder();
csv.AppendLine("MaxSideLen,FileName,DetectionCount,ElapsedMilliseconds,RapidDetectMilliseconds,RawText,Error");
foreach (var r in rows)
{
    csv.AppendLine(string.Join(",",
        r.MaxSideLen,
        Csv(r.FileName),
        r.DetectionCount,
        F(r.ElapsedMilliseconds),
        F(r.RapidDetectMilliseconds),
        Csv(r.RawText),
        Csv(r.Error)));
}
await File.WriteAllTextAsync(resultsCsv, csv.ToString(), Encoding.UTF8);

string summaryCsv = Path.Combine(rootOutput, "SweepSummary.csv");
var summary = new StringBuilder();
summary.AppendLine("MaxSideLen,Images,Succeeded,Failed,AverageMilliseconds,MedianMilliseconds,P95Milliseconds,MaxMilliseconds");
foreach (int len in maxSideLens)
{
    var group = rows.Where(r => r.MaxSideLen == len).ToList();
    var ok = group.Where(r => string.IsNullOrWhiteSpace(r.Error)).Select(r => r.ElapsedMilliseconds).OrderBy(x => x).ToList();
    double avg = ok.Count > 0 ? ok.Average() : 0;
    double median = Percentile(ok, 0.50);
    double p95 = Percentile(ok, 0.95);
    double max = ok.Count > 0 ? ok.Max() : 0;
    summary.AppendLine(string.Join(",", len, group.Count, ok.Count, group.Count - ok.Count, F(avg), F(median), F(p95), F(max)));
    Console.WriteLine($"{len}: {ok.Count}/{group.Count} succeeded | avg {avg:F0} ms | median {median:F0} ms | p95 {p95:F0} ms");
}
await File.WriteAllTextAsync(summaryCsv, summary.ToString(), Encoding.UTF8);

Console.WriteLine();
Console.WriteLine("Sweep complete.");
Console.WriteLine(rootOutput);
Console.WriteLine("Created SweepResults.csv, SweepSummary.csv, and RawText folders for each setting.");

static double Percentile(List<double> sorted, double p)
{
    if (sorted.Count == 0) return 0;
    if (sorted.Count == 1) return sorted[0];
    double index = (sorted.Count - 1) * p;
    int lo = (int)Math.Floor(index);
    int hi = (int)Math.Ceiling(index);
    if (lo == hi) return sorted[lo];
    double frac = index - lo;
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

static string Csv(string? value)
{
    value ??= "";
    return "\"" + value.Replace("\"", "\"\"") + "\"";
}

static string F(double value) => value.ToString("0.###", CultureInfo.InvariantCulture);

class SweepRow
{
    public int MaxSideLen { get; set; }
    public string FileName { get; set; } = "";
    public int DetectionCount { get; set; }
    public double ElapsedMilliseconds { get; set; }
    public double RapidDetectMilliseconds { get; set; }
    public string RawText { get; set; } = "";
    public string Error { get; set; } = "";
}
