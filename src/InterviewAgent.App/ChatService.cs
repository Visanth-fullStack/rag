using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace InterviewAgent.App
{
    public class ChatService : IDisposable
    {
        private readonly HttpClient _httpClient;
        private CancellationTokenSource _cts;

        public event Action<string, string> OnMessageReceived; // response, query
        public event Action<string> OnStatusChanged;

        public ChatService()
        {
            _httpClient = new HttpClient();
            _httpClient.Timeout = Timeout.InfiniteTimeSpan;
        }

        public async Task SendMessageAsync(string query)
        {
            // Cancel previous streaming if exists
            _cts?.Cancel();
            _cts = new CancellationTokenSource();
            
            OnStatusChanged?.Invoke("Thinking...");
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var payload = JsonConvert.SerializeObject(new { query = query });
                    var content = new StringContent(payload, Encoding.UTF8, "application/json");

                    using var request = new HttpRequestMessage(HttpMethod.Post, "http://localhost:8000/chat/stream");
                    request.Content = content;

                    using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, _cts.Token);
                    response.EnsureSuccessStatusCode();

                    using var stream = await response.Content.ReadAsStreamAsync();
                    using var reader = new StreamReader(stream);

                    while (!reader.EndOfStream && !_cts.Token.IsCancellationRequested)
                    {
                        var line = await reader.ReadLineAsync();
                        if (string.IsNullOrWhiteSpace(line)) continue;

                        if (line.StartsWith("data: "))
                        {
                            var json = line.Substring(6);
                            var data = JsonConvert.DeserializeObject<dynamic>(json);

                            if (data.error != null)
                            {
                                OnMessageReceived?.Invoke($"Error: {data.error}", "");
                            }
                            else
                            {
                                string generation = data.generation;
                                string resQuery = data.query;
                                OnMessageReceived?.Invoke(generation, resQuery);
                            }
                        }
                    }
                    OnStatusChanged?.Invoke("Done.");
                }
                catch (TaskCanceledException)
                {
                    OnStatusChanged?.Invoke("Canceled.");
                }
                catch (Exception ex)
                {
                    OnStatusChanged?.Invoke($"Error: {ex.Message}");
                }
            }, _cts.Token);
        }

        public void Disconnect()
        {
            _cts?.Cancel();
        }

        public void Dispose()
        {
            Disconnect();
            _httpClient?.Dispose();
        }
    }
}
