using System;
using System.Collections.ObjectModel;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace InterviewAgent.App
{
    public class ChatMessage
    {
        public string Sender { get; set; }
        public string Message { get; set; }
        public HorizontalAlignment Alignment { get; set; }
        public Brush BackgroundColor { get; set; }
    }

    public partial class MainWindow : Window
    {
        private AudioTranscriber _micTranscriber;
        private AudioTranscriber _systemTranscriber;
        private ChatService _chatService;
        private StringBuilder _fullTranscription = new StringBuilder();
        private ObservableCollection<ChatMessage> _chatMessages = new ObservableCollection<ChatMessage>();

        public MainWindow()
        {
            InitializeComponent();
            
            _micTranscriber = new AudioTranscriber();
            _systemTranscriber = new AudioTranscriber();
            _chatService = new ChatService();
            
            _micTranscriber.OnTranscriptionReceived += HandleTranscription;
            _systemTranscriber.OnTranscriptionReceived += HandleTranscription;

            _micTranscriber.OnStatusChanged += (s) => Dispatcher.Invoke(() => StatusText.Text = $"Mic: {s}");
            _systemTranscriber.OnStatusChanged += (s) => Dispatcher.Invoke(() => StatusText.Text += $" | System: {s}");

            _chatService.OnMessageReceived += HandleChatMessage;
            _chatService.OnStatusChanged += (s) => Dispatcher.Invoke(() => StatusText.Text += $" | Chat: {s}");

            ChatItemsControl.ItemsSource = _chatMessages;
        }

        private void HandleTranscription(string text, string source)
        {
            Dispatcher.Invoke(() =>
            {
                string tag = source == "mic" ? "[ME]" : "[SYSTEM]";
                _fullTranscription.AppendLine($"{tag}: {text}");
                TranscriptionText.Text = _fullTranscription.ToString();
                TranscriptionText.Foreground = Brushes.Black;
                TranscriptionScrollViewer.ScrollToEnd();
            });
        }

        private void HandleChatMessage(string response, string query)
        {
            Dispatcher.Invoke(() =>
            {
                if (_chatMessages.Count > 0 && _chatMessages[_chatMessages.Count - 1].Sender == "AI Assistant")
                {
                    var lastMessage = _chatMessages[_chatMessages.Count - 1];
                    // Replace the item to trigger UI update
                    _chatMessages[_chatMessages.Count - 1] = new ChatMessage
                    {
                        Sender = "AI Assistant",
                        Message = lastMessage.Message + response,
                        Alignment = lastMessage.Alignment,
                        BackgroundColor = lastMessage.BackgroundColor
                    };
                }
                else
                {
                    _chatMessages.Add(new ChatMessage 
                    { 
                        Sender = "AI Assistant", 
                        Message = response, 
                        Alignment = HorizontalAlignment.Left,
                        BackgroundColor = new SolidColorBrush(Color.FromRgb(240, 240, 240))
                    });
                }
                ChatScrollViewer.ScrollToEnd();
            });
        }

        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            StartButton.IsEnabled = false;
            StopButton.IsEnabled = true;
            
            _fullTranscription.Clear();
            TranscriptionText.Text = "Listening to both Mic and System...";
            StatusText.Text = "Starting...";

            try 
            {
                var micTask = _micTranscriber.StartAsync("mic");
                var systemTask = _systemTranscriber.StartAsync("system");
                await Task.WhenAll(micTask, systemTask);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}");
                StopButton_Click(null, null);
            }
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            _micTranscriber.Stop();
            _systemTranscriber.Stop();
            StartButton.IsEnabled = true;
            StopButton.IsEnabled = false;
            StatusText.Text = "Stopped";
        }

        private void ChatToggleButton_Click(object sender, RoutedEventArgs e)
        {
            if (ChatPanel.Visibility == Visibility.Collapsed)
            {
                ChatPanel.Visibility = Visibility.Visible;
                ChatColumn.Width = new GridLength(300);
                ChatToggleButton.Content = "Close Chat";
            }
            else
            {
                ChatPanel.Visibility = Visibility.Collapsed;
                ChatColumn.Width = new GridLength(0);
                ChatToggleButton.Content = "Open Chat";
            }
        }

        private async void SendChat_Click(object sender, RoutedEventArgs e)
        {
            await SendMessage();
        }

        private async void ChatInput_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                await SendMessage();
            }
        }

        private async Task SendMessage()
        {
            string query = ChatInput.Text.Trim();
            if (string.IsNullOrEmpty(query)) return;

            _chatMessages.Add(new ChatMessage 
            { 
                Sender = "You", 
                Message = query, 
                Alignment = HorizontalAlignment.Right,
                BackgroundColor = new SolidColorBrush(Color.FromRgb(220, 235, 255))
            });

            ChatInput.Clear();
            ChatScrollViewer.ScrollToEnd();

            try
            {
                await _chatService.SendMessageAsync(query);
            }
            catch (Exception ex)
            {
                _chatMessages.Add(new ChatMessage { Sender = "System", Message = $"Error: {ex.Message}" });
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            _micTranscriber.Dispose();
            _systemTranscriber.Dispose();
            _chatService.Dispose();
            base.OnClosed(e);
        }
    }
}
