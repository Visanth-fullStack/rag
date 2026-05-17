using System;
using System.Net.WebSockets;
using System.Threading;
using System.Threading.Tasks;
using System.Text;
using NAudio.Wave;
using NAudio.CoreAudioApi;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;

namespace InterviewAgent.App
{
    public class AudioTranscriber : IDisposable
    {
        private ClientWebSocket _webSocket;
        private WasapiCapture _capture;
        private CancellationTokenSource _cts;
        private string _sourceName;
        
        public event Action<string, string> OnTranscriptionReceived; // text, source
        public event Action<string> OnStatusChanged;

        public async Task StartAsync(string sourceName)
        {
            _sourceName = sourceName;
            _cts = new CancellationTokenSource();
            _webSocket = new ClientWebSocket();
            
            OnStatusChanged?.Invoke($"Connecting {sourceName}...");
            try 
            {
                await _webSocket.ConnectAsync(new Uri($"ws://localhost:8000/ws/transcribe/{sourceName}"), _cts.Token);
                OnStatusChanged?.Invoke($"{sourceName} Connected.");
            }
            catch (Exception ex)
            {
                OnStatusChanged?.Invoke($"{sourceName} failed: {ex.Message}");
                return;
            }

            if (sourceName == "system")
                _capture = new WasapiLoopbackCapture();
            else
                _capture = new WasapiCapture();

            var targetFormat = new WaveFormat(16000, 16, 1);
            
            _capture.DataAvailable += async (s, e) =>
            {
                if (_webSocket.State == WebSocketState.Open)
                {
                    byte[] buffer = e.Buffer;
                    int bytesRecorded = e.BytesRecorded;

                    if (!_capture.WaveFormat.Equals(targetFormat))
                    {
                        using (var ms = new MemoryStream(e.Buffer, 0, e.BytesRecorded))
                        using (var reader = new RawSourceWaveStream(ms, _capture.WaveFormat))
                        using (var resampler = new MediaFoundationResampler(reader, targetFormat))
                        {
                            var resampledBuffer = new byte[e.BytesRecorded];
                            int read = resampler.Read(resampledBuffer, 0, resampledBuffer.Length);
                            buffer = resampledBuffer;
                            bytesRecorded = read;
                        }
                    }

                    if (bytesRecorded > 0)
                    {
                        await _webSocket.SendAsync(new ArraySegment<byte>(buffer, 0, bytesRecorded), 
                            WebSocketMessageType.Binary, true, _cts.Token);
                    }
                }
            };

            _capture.StartRecording();
            _ = ReceiveLoop();
        }

        private async Task ReceiveLoop()
        {
            var buffer = new byte[1024 * 4];
            while (_webSocket.State == WebSocketState.Open && !_cts.Token.IsCancellationRequested)
            {
                try {
                    var result = await _webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), _cts.Token);
                    if (result.MessageType == WebSocketMessageType.Close) break;

                    var json = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    var data = JsonConvert.DeserializeObject<dynamic>(json);
                    string text = data.text;
                    string source = data.source;
                    
                    OnTranscriptionReceived?.Invoke(text, source);
                } catch { break; }
            }
        }

        public void Stop()
        {
            _cts?.Cancel();
            _capture?.StopRecording();
            _capture?.Dispose();
            _webSocket?.Dispose();
        }

        public void Dispose() => Stop();
    }
}
