from move_comparison import compare_positions
benchmark_video = './dance_videos/benchmarkOdissi.mp4'
# For live webcam feed, simply call the function with the benchmark video only
error, acc, n = compare_positions(benchmark_video)
print("Overall Error:", error/n)
print("Overall Accuracy:", acc/n)
