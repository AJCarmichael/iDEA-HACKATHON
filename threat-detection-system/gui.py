import tkinter as tk
from tkinter import scrolledtext
from detector import run_detections
from reasoner import get_reasoning

def scan_system():
    findings = run_detections()
    reasoning = get_reasoning(findings)
    
    text_area.delete(1.0, tk.END)
    
    text_area.insert(tk.END, "=== Suspicious Processes ===\n")
    if findings['suspicious_processes']:
        for proc in findings['suspicious_processes']:
            text_area.insert(tk.END, f"PID {proc['pid']} - {proc['name']}\n")
    else:
        text_area.insert(tk.END, "None found\n")
    
    text_area.insert(tk.END, "\n=== Suspicious Network Connections ===\n")
    if findings['suspicious_connections']:
        for conn in findings['suspicious_connections']:
            text_area.insert(tk.END, f"PID {conn['pid']} - {conn['raddr']} ({conn['status']})\n")
    else:
        text_area.insert(tk.END, "None found\n")
    
    text_area.insert(tk.END, "\n=== AI Reasoning ===\n")
    text_area.insert(tk.END, reasoning)

def run_gui():
    global text_area
    window = tk.Tk()
    window.title("Insider Threat Detector")
    window.geometry("700x500")
    
    scan_button = tk.Button(window, text="Scan System", command=scan_system)
    scan_button.pack(pady=10)
    
    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=80, height=25)
    text_area.pack(padx=10, pady=10)
    
    window.mainloop()

if __name__ == '__main__':
    run_gui()
