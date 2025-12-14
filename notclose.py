"""
=============================================================================
VISUAL MATHEMATICAL JOURNEY THROUGH AI ARCHITECTURE EVOLUTION
SageMath 10.7 Interactive Notebook - Complete Implementation
=============================================================================

Educational Resource: Understanding AI through Geometric Mathematics
Covers: ResNet, Transformers, LSTM, Segmentation, Efficient Architectures

Execute sections sequentially. Each includes:
- Mathematical foundations
- Geometric interpretations  
- Interactive visualizations
- Implementation insights
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Wedge, Polygon
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print("="*80)
print("AI EVOLUTION: VISUAL MATHEMATICAL JOURNEY")
print("="*80)
print("\nSections:")
print("1. Convolution - Feature Extraction")
print("2. LSTM - Sequential Memory")  
print("3. ResNet - Residual Learning")
print("4. Transformer - Attention Mechanism")
print("5. Image Segmentation - Pixel-wise Prediction")
print("6. Efficient Architectures - Less is More")
print("\nExecute: visualize_all_architectures()")
print("="*80 + "\n")

#==============================================================================
# SECTION 1: CONVOLUTION FUNDAMENTALS
#==============================================================================

def visualize_convolution_math():
    """Convolution: Y[i,j] = Σ_m Σ_n X[i+m,j+n] * K[m,n]"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('CONVOLUTION: Local Feature Extraction', 
                 fontsize=18, fontweight='bold')
    
    np.random.seed(42)
    input_map = np.random.randn(8, 8)
    
    kernels = {
        'Edge Detection': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
        'Blur (Smoothing)': np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16,
        'Sharpen': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    }
    
    for idx, (name, kernel) in enumerate(kernels.items()):
        im1 = axes[0,idx].imshow(input_map, cmap='viridis')
        axes[0,idx].set_title(f'Input (8×8)', fontweight='bold', fontsize=13)
        axes[0,idx].grid(True, color='white', lw=0.5, alpha=0.3)
        plt.colorbar(im1, ax=axes[0,idx], fraction=0.046)
        
        output = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                output[i,j] = np.sum(input_map[i:i+3,j:j+3] * kernel)
        
        im2 = axes[1,idx].imshow(output, cmap='coolwarm')
        axes[1,idx].set_title(f'{name}\n→ Output (6×6)', fontweight='bold', fontsize=13)
        plt.colorbar(im2, ax=axes[1,idx], fraction=0.046)
        
        formula = f"Kernel:\n{kernel[1,1]:.2f}"
        axes[1,idx].text(0.02, 0.98, formula, transform=axes[1,idx].transAxes,
                        fontsize=10, va='top',
                        bbox=dict(boxstyle='round', fc='yellow', alpha=0.9))
    
    plt.tight_layout()
    return fig

#==============================================================================
# SECTION 2: LSTM ARCHITECTURE
#==============================================================================

def visualize_lstm_complete():
    """LSTM: Gated memory cells for long-term dependencies"""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35)
    fig.suptitle('LSTM: Long Short-Term Memory Networks', 
                 fontsize=18, fontweight='bold')
    
    # Cell structure
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('LSTM Cell Gates', fontweight='bold', fontsize=14)
    
    ax1.plot([1, 9], [9, 9], 'purple', lw=4, alpha=0.7, label='Cell State C_t')
    ax1.arrow(8.7, 9, 0.2, 0, head_width=0.25, head_length=0.15, fc='purple', ec='purple')
    ax1.plot([1, 9], [2, 2], 'blue', lw=3, alpha=0.7, label='Hidden h_t')
    ax1.arrow(8.7, 2, 0.2, 0, head_width=0.25, head_length=0.15, fc='blue', ec='blue')
    
    ax1.arrow(3, 0.3, 0, 1.3, head_width=0.2, head_length=0.15, fc='green', ec='green', lw=2)
    ax1.text(3, 0.1, 'x_t', ha='center', fontsize=13, fontweight='bold')
    
    gates = [
        (3, 8.3, 'σ', 'Forget', 'coral'),
        (4.5, 6.5, 'σ', 'Input', 'lightblue'),
        (6, 6.5, 'tanh', 'Cell', 'lightgreen'),
        (7.5, 3.5, 'σ', 'Output', 'plum')
    ]
    
    for x, y, sym, label, color in gates:
        gate = FancyBboxPatch((x-0.4, y), 0.8, 1, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', lw=2)
        ax1.add_patch(gate)
        ax1.text(x, y+0.5, sym, ha='center', va='center', 
                fontsize=14 if 'tanh' in sym else 18, fontweight='bold')
        ax1.text(x, y-0.4, label, ha='center', fontsize=9, fontweight='bold')
    
    ops = [(4, 9, '×'), (5.5, 9, '+'), (6.5, 9, '×'), (7.5, 2.5, '×')]
    for x, y, op in ops:
        circle = Circle((x, y), 0.35, fc='yellow', ec='black', lw=2)
        ax1.add_patch(circle)
        ax1.text(x, y, op, ha='center', va='center', fontsize=18, fontweight='bold')
    
    ax1.legend(loc='upper left', fontsize=10)
    
    eq = ("Gates:\nf_t = σ(W_f[h,x]+b_f)\ni_t = σ(W_i[h,x]+b_i)\n"
          "C̃_t = tanh(W_C[h,x]+b_C)\nC_t = f_t⊙C_{t-1}+i_t⊙C̃_t\n"
          "o_t = σ(W_o[h,x]+b_o)\nh_t = o_t⊙tanh(C_t)")
    ax1.text(0.3, 5, eq, fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.95))
    
    # Unrolled
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_title('Unrolled Through Time', fontweight='bold', fontsize=13)
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    
    for t in range(6):
        x = 1 + t * 1.8
        cell = FancyBboxPatch((x-0.35, 3), 0.7, 1.5, boxstyle="round,pad=0.08",
                             facecolor='lightcyan', edgecolor='navy', lw=2)
        ax2.add_patch(cell)
        ax2.text(x, 3.75, f'LSTM\nt={t}', ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax2.arrow(x, 2, 0, 0.8, head_width=0.15, head_length=0.12, fc='green', ec='green')
        ax2.text(x, 1.5, f'x_{t}', ha='center', fontsize=10)
        ax2.arrow(x, 4.5, 0, 0.8, head_width=0.15, head_length=0.12, fc='red', ec='red')
        ax2.text(x, 5.5, f'h_{t}', ha='center', fontsize=10)
        
        if t < 5:
            ax2.arrow(x+0.35, 3.75, 1.05, 0, head_width=0.15, head_length=0.12,
                     fc='blue', ec='blue', linestyle='--', alpha=0.7)
    
    # Activations
    ax3 = fig.add_subplot(gs[1, 1:])
    x = np.linspace(-6, 6, 200)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    
    ax3.plot(x, sigmoid, 'b-', lw=3, label='σ(x) = 1/(1+e^(-x))', alpha=0.8)
    ax3.plot(x, tanh, 'r-', lw=3, label='tanh(x)', alpha=0.8)
    ax3.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax3.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax3.fill_between(x, sigmoid, alpha=0.2)
    ax3.fill_between(x, tanh, alpha=0.2)
    ax3.grid(True, alpha=0.4)
    ax3.legend(fontsize=11)
    ax3.set_xlabel('Input', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Output', fontsize=12, fontweight='bold')
    ax3.set_title('Activation Functions', fontweight='bold', fontsize=13)
    
    # State evolution
    ax4 = fig.add_subplot(gs[2:, :])
    T = 100
    C = np.zeros(T)
    h = np.zeros(T)
    f_gate = np.zeros(T)
    i_gate = np.zeros(T)
    
    np.random.seed(42)
    for t in range(1, T):
        f_gate[t] = 0.85 + 0.1 * np.sin(t/10)
        i_gate[t] = 0.3 * np.random.rand()
        C[t] = f_gate[t] * C[t-1] + i_gate[t] * np.sin(t/8) + 0.05*np.random.randn()
        h[t] = 0.9 * np.tanh(C[t])
    
    ax4.plot(C, lw=2.5, label='Cell State C_t', color='purple', alpha=0.8)
    ax4.plot(h, lw=2.5, label='Hidden State h_t', color='blue', alpha=0.8)
    ax4.plot(f_gate, lw=1.5, label='Forget Gate f_t', color='coral', ls='--', alpha=0.7)
    ax4.plot(i_gate, lw=1.5, label='Input Gate i_t', color='green', ls='--', alpha=0.7)
    
    ax4.fill_between(range(T), C, alpha=0.15, color='purple')
    ax4.grid(True, alpha=0.4)
    ax4.legend(fontsize=11, loc='upper right')
    ax4.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax4.set_ylabel('State Value', fontsize=12, fontweight='bold')
    ax4.set_title('LSTM Memory Evolution', fontweight='bold', fontsize=14)
    ax4.axhline(0, color='k', lw=0.5, alpha=0.5)
    
    plt.tight_layout()
    return fig

#==============================================================================
# SECTION 3: RESNET RESIDUAL LEARNING
#==============================================================================

def visualize_resnet_complete():
    """ResNet: y = F(x) + x enables deep networks"""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35)
    fig.suptitle('ResNet: Residual Learning Framework', 
                 fontsize=18, fontweight='bold')
    
    # Residual block
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 14)
    ax1.axis('off')
    ax1.set_title('Residual Block', fontweight='bold', fontsize=14)
    
    input_box = FancyBboxPatch((4, 1), 2, 1, boxstyle="round,pad=0.1",
                               facecolor='lightgreen', edgecolor='black', lw=2.5)
    ax1.add_patch(input_box)
    ax1.text(5, 1.5, 'x\nInput', ha='center', va='center', fontsize=12, fontweight='bold')
    
    conv1 = FancyBboxPatch((4, 3.5), 2, 1.3, boxstyle="round,pad=0.1",
                           facecolor='lightblue', edgecolor='black', lw=2)
    ax1.add_patch(conv1)
    ax1.text(5, 4.15, 'Conv+BN\n+ReLU', ha='center', va='center', fontsize=10, fontweight='bold')
    
    conv2 = FancyBboxPatch((4, 6), 2, 1.3, boxstyle="round,pad=0.1",
                           facecolor='lightblue', edgecolor='black', lw=2)
    ax1.add_patch(conv2)
    ax1.text(5, 6.65, 'Conv+BN', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax1.arrow(5, 2, 0, 1.3, head_width=0.25, head_length=0.15, fc='black', ec='black', lw=2.5)
    ax1.arrow(5, 4.8, 0, 1, head_width=0.25, head_length=0.15, fc='black', ec='black', lw=2.5)
    ax1.text(5.5, 3, 'F(x)', fontsize=11, fontweight='bold', color='darkblue')
    
    # Skip connection
    ax1.annotate('', xy=(7.5, 9.5), xytext=(7.5, 2),
                arrowprops=dict(arrowstyle='->', lw=4, color='red', 
                              connectionstyle='arc3,rad=0.4'))
    ax1.text(8.5, 5.5, 'Skip\nIdentity', ha='center', va='center',
            fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.9))
    
    add = Circle((5, 9.5), 0.6, fc='yellow', ec='black', lw=2.5)
    ax1.add_patch(add)
    ax1.text(5, 9.5, '+', ha='center', va='center', fontsize=24, fontweight='bold')
    ax1.arrow(5, 7.3, 0, 1.6, head_width=0.25, head_length=0.15, fc='black', ec='black', lw=2.5)
    
    relu = FancyBboxPatch((4, 10.8), 2, 0.9, boxstyle="round,pad=0.1",
                          facecolor='orange', edgecolor='black', lw=2)
    ax1.add_patch(relu)
    ax1.text(5, 11.25, 'ReLU', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.arrow(5, 10.1, 0, 0.5, head_width=0.25, head_length=0.12, fc='black', ec='black', lw=2.5)
    
    ax1.arrow(5, 11.7, 0, 0.8, head_width=0.25, head_length=0.15, fc='green', ec='green', lw=2.5)
    ax1.text(5, 13, 'y=F(x)+x', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.9))
    
    math = ("y = F(x,{W_i}) + x\n\nF: residual function\nx: identity mapping\n\n"
            "Key: Learning F(x)\nis easier than H(x)")
    ax1.text(0.8, 6, math, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.95, ec='orange', lw=2))
    
    # Architecture
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_title('ResNet-50 Architecture', fontweight='bold', fontsize=13)
    ax2.set_xlim(0, 17)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    stages = [
        ('Input\n224×224', 'lightgreen', 8),
        ('Conv1\n112×112', 'lightblue', 7.5),
        ('Stage1\n56×56', 'lightcoral', 7),
        ('Stage2\n28×28', 'plum', 6),
        ('Stage3\n14×14', 'lightyellow', 5),
        ('Stage4\n7×7', 'lightcyan', 4),
        ('Pool\n1×1', 'wheat', 3),
        ('FC', 'orange', 2.5)
    ]
    
    x_pos = np.linspace(1.5, 15.5, len(stages))
    
    for i, ((label, color, height), x) in enumerate(zip(stages, x_pos)):
        w = 1.4
        box = FancyBboxPatch((x-w/2, 5-height/2), w, height,
                            boxstyle="round,pad=0.08",
                            facecolor=color, edgecolor='black', lw=2)
        ax2.add_patch(box)
        ax2.text(x, 5, label, ha='center', va='center', fontsize=8.5, fontweight='bold')
        
        if i < len(stages)-1:
            ax2.arrow(x+w/2+0.1, 5, x_pos[i+1]-x-w-0.3, 0,
                     head_width=0.4, head_length=0.2, fc='gray', ec='gray', lw=2)
        
        if 2 <= i <= 5:
            ax2.annotate('', xy=(x_pos[i+1]-w/2-0.1, 5), xytext=(x+w/2+0.1, 5),
                        arrowprops=dict(arrowstyle='->', lw=2.5, color='red',
                                      ls='--', alpha=0.7, connectionstyle='arc3,rad=0.6'))
    
    # Gradient flow
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.set_title('Gradient Flow Comparison', fontweight='bold', fontsize=13)
    
    layers = np.arange(1, 101)
    np.random.seed(42)
    plain = np.exp(-layers/12) * (1 + 0.3*np.random.randn(len(layers)))
    resnet = np.exp(-layers/80) * (1 + 0.15*np.random.randn(len(layers)))
    plain = np.maximum(plain, 1e-10)
    resnet = np.maximum(resnet, 1e-10)
    
    ax3.semilogy(layers, plain, 'r-', lw=2.5, label='Plain Network', alpha=0.8)
    ax3.semilogy(layers, resnet, 'b-', lw=2.5, label='ResNet', alpha=0.8)
    ax3.axhline(y=0.01, color='orange', ls='--', label='Threshold', lw=2)
    ax3.fill_between(layers, plain, 1e-10, alpha=0.2, color='red')
    ax3.fill_between(layers, resnet, 1e-10, alpha=0.2, color='blue')
    ax3.grid(True, alpha=0.4, which='both')
    ax3.legend(fontsize=11)
    ax3.set_xlabel('Layer Depth', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Gradient Magnitude (log)', fontsize=12, fontweight='bold')
    
    # Learning curves
    ax4 = fig.add_subplot(gs[2, :2])
    epochs = np.arange(1, 91)
    
    plain18 = 1 - 0.7*np.exp(-epochs/20) + 0.02*np.random.randn(len(epochs))
    plain50 = 1 - 0.5*np.exp(-epochs/20) + 0.03*np.random.randn(len(epochs))
    res18 = 1 - 0.75*np.exp(-epochs/15) + 0.015*np.random.randn(len(epochs))
    res50 = 1 - 0.8*np.exp(-epochs/15) + 0.015*np.random.randn(len(epochs))
    
    ax4.plot(epochs, plain18, 'r--', lw=2, label='Plain-18', alpha=0.7)
    ax4.plot(epochs, plain50, 'r:', lw=2, label='Plain-50', alpha=0.7)
    ax4.plot(epochs, res18, 'b-', lw=2.5, label='ResNet-18', alpha=0.8)
    ax4.plot(epochs, res50, 'g-', lw=2.5, label='ResNet-50', alpha=0.8)
    
    ax4.grid(True, alpha=0.4)
    ax4.legend(fontsize=11)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Training Performance', fontweight='bold', fontsize=13)
    
    # Loss landscape
    ax5 = fig.add_subplot(gs[2:, 2], projection='3d')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    Z_plain = np.sin(2*X)*np.sin(2*Y) + 0.1*(X**2 + Y**2)
    Z_resnet = 0.3*(X**2 + Y**2) + 0.05*np.sin(4*X)*np.sin(4*Y)
    
    ax5.plot_surface(X, Y, Z_plain, cmap='Reds', alpha=0.6)
    ax5.plot_surface(X, Y, Z_resnet, cmap='Blues', alpha=0.6)
    
    ax5.set_xlabel('θ1', fontsize=10, fontweight='bold')
    ax5.set_ylabel('θ2', fontsize=10, fontweight='bold')
    ax5.set_zlabel('Loss', fontsize=10, fontweight='bold')
    ax5.set_title('Loss Landscape\nRed: Plain | Blue: ResNet', fontweight='bold', fontsize=11)
    ax5.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    return fig

#==============================================================================
# SECTION 4: TRANSFORMER ATTENTION
#==============================================================================

def visualize_transformer_complete():
    """Transformer: Attention(Q,K,V) = softmax(QK^T/√d_k)V"""
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.4)
    fig.suptitle('TRANSFORMER: Attention Mechanism', 
                 fontsize=18, fontweight='bold')
    
    # Architecture
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 18)
    ax1.axis('off')
    ax1.set_title('Transformer Block', fontweight='bold', fontsize=14)
    
    # Encoder
    enc = [
        (2, 1, 2.5, 1, 'Embedding', 'lightgreen'),
        (2, 2.5, 2.5, 0.7, '+PosEnc', 'wheat'),
        (2, 3.7, 2.5, 1.5, 'Multi-Head\nAttention', 'lightblue'),
        (2, 5.6, 2.5, 0.6, 'Add&Norm', 'lightyellow'),
        (2, 6.6, 2.5, 1.2, 'Feed\nForward', 'lightcoral'),
        (2, 8.2, 2.5, 0.6, 'Add&Norm', 'lightyellow'),
    ]
    
    for x, y, w, h, label, color in enc:
        box = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.08",
                            facecolor=color, edgecolor='black', lw=2)
        ax1.add_patch(box)
        ax1.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Decoder  
    dec = [
        (5.5, 1, 2.5, 1, 'Embedding', 'lightgreen'),
        (5.5, 2.5, 2.5, 0.7, '+PosEnc', 'wheat'),
        (5.5, 3.7, 2.5, 1.5, 'Masked\nAttention', 'plum'),
        (5.5, 5.6, 2.5, 0.6, 'Add&Norm', 'lightyellow'),
        (5.5, 6.6, 2.5, 1.5, 'Cross\nAttention', 'lightcyan'),
        (5.5, 8.5, 2.5, 0.6, 'Add&Norm', 'lightyellow'),
        (5.5, 9.5, 2.5, 1.2, 'Feed\nForward', 'lightcoral'),
        (5.5, 11.1, 2.5, 0.6, 'Add&Norm', 'lightyellow'),
        (5.5, 12.2, 2.5, 0.9, 'Linear+\nSoftmax', 'orange'),
    ]
    
    for x, y, w, h, label, color in dec:
        box = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.08",
                            facecolor=color, edgecolor='black', lw=2)
        ax1.add_patch(box)
        ax1.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    for y_start, y_end in [(2, 3.5), (5.2, 5.4), (6.2, 6.4), (7.8, 8)]:
        ax1.arrow(3.25, y_start, 0, y_end-y_start-0.2, 
                 head_width=0.2, head_length=0.12, fc='k', ec='k', lw=2)
    
    for y_start, y_end in [(2, 3.5), (5.2, 5.4), (6.2, 6.4), (8.1, 8.3), (9.1, 9.3), (10.7, 10.9), (11.7, 12)]:
        ax1.arrow(6.75, y_start, 0, y_end-y_start-0.2, 
                 head_width=0.2, head_length=0.12, fc='k', ec='k', lw=2)
    
    # Cross-attention connection
    ax1.annotate('K,V', xy=(5.5, 7.3), xytext=(4.5, 8.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='blue',
                              connectionstyle='arc3,rad=0.3'))
    
    # Skip connections
    for y1, y2 in [(3.2, 5.9), (6.2, 8.5)]:
        ax1.annotate('', xy=(1.8, y2), xytext=(1.8, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red', ls='--'))
    
    ax1.text(3.25, 0.3, 'ENCODER', ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', fc='lightblue'))
    ax1.text(6.75, 0.3, 'DECODER', ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', fc='lightgreen'))
    
    # Attention math
    ax2 = fig.add_subplot(gs[0, 1:3])
    ax2.set_title('Self-Attention Mathematics', fontweight='bold', fontsize=13)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    seq_len, d = 6, 4
    np.random.seed(42)
    Q = np.random.randn(seq_len, d) * 0.5
    K = np.random.randn(seq_len, d) * 0.5
    V = np.random.randn(seq_len, d) * 0.5
    
    scores = np.dot(Q, K.T) / np.sqrt(d)
    attn = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    
    positions = [1.5, 5, 8.5]
    labels = ['Q', 'K', 'V']
    matrices = [Q, K, V]
    cmaps = ['Reds', 'Blues', 'Greens']
    
    for pos, label, mat, cmap in zip(positions, labels, matrices, cmaps):
        box = FancyBboxPatch((pos-0.7, 3), 1.4, 4, boxstyle="round,pad=0.05",
                            facecolor='white', edgecolor='black', lw=2)
        ax2.add_patch(box)
        ax2.imshow(mat, cmap=cmap, aspect='auto', 
                  extent=[pos-0.6, pos+0.6, 3.2, 6.8], alpha=0.7)
        ax2.text(pos, 7.3, label, ha='center', fontsize=11, fontweight='bold')
        ax2.text(pos, 2.5, f'{seq_len}×{d}', ha='center', fontsize=9)
    
    formula = ("Attention(Q,K,V)=\nsoftmax(QK^T/√d_k)V\n\n"
              "Steps:\n1. QK^T scores\n2. Scale by √d_k\n"
              "3. Softmax\n4. Weight V")
    ax2.text(10.5, 5, formula, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.95, ec='orange', lw=2),
            va='center')
    
    # Attention weights
    ax3 = fig.add_subplot(gs[1, 1:3])
    im = ax3.imshow(attn, cmap='YlOrRd', aspect='auto')
    ax3.set_title('Attention Weight Matrix', fontweight='bold', fontsize=13)
    ax3.set_xlabel('Key Position', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Query Position', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Weight', fraction=0.046)
    
    for i in range(seq_len):
        for j in range(seq_len):
            color = 'white' if attn[i,j] > 0.5 else 'black'
            ax3.text(j, i, f'{attn[i,j]:.2f}', ha='center', va='center',
                    color=color, fontsize=8, fontweight='bold')
    
    # Multi-head
    ax4 = fig.add_subplot(gs[0:2, 3])
    ax4.set_title('Multi-Head\nAttention', fontweight='bold', fontsize=12)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 12)
    ax4.axis('off')
    
    input_b = FancyBboxPatch((3.5, 0.5), 3, 0.8, boxstyle="round,pad=0.05",
                             facecolor='lightgreen', edgecolor='black', lw=2)
    ax4.add_patch(input_b)
    ax4.text(5, 0.9, 'Input', ha='center', fontsize=10, fontweight='bold')
    
    heads = 4
    head_colors = plt.cm.Set3(range(heads))
    
    for h in range(heads):
        x = 1 + h * 2
        
        proj = FancyBboxPatch((x, 2.5), 1.5, 1, boxstyle="round,pad=0.05",
                             facecolor=head_colors[h], edgecolor='black', lw=1.5)
        ax4.add_patch(proj)
        ax4.text(x+0.75, 3, f'Head{h+1}\nQ,K,V', ha='center', fontsize=8, fontweight='bold')
        
        attn_b = FancyBboxPatch((x, 4.5), 1.5, 1, boxstyle="round,pad=0.05",
                               facecolor=head_colors[h], edgecolor='black', lw=1.5, alpha=0.7)
        ax4.add_patch(attn_b)
        ax4.text(x+0.75, 5, 'Attn', ha='center', fontsize=7, fontweight='bold')
        
        ax4.arrow(x+0.75, 1.3, 0, 1, head_width=0.15, head_length=0.1, fc='gray', ec='gray')
        ax4.arrow(x+0.75, 3.5, 0, 0.8, head_width=0.15, head_length=0.1, fc='gray', ec='gray')
    
    concat = FancyBboxPatch((2, 7), 6, 0.8, boxstyle="round,pad=0.05",
                            facecolor='lightyellow', edgecolor='black', lw=2)
    ax4.add_patch(concat)
    ax4.text(5, 7.4, 'Concatenate', ha='center', fontsize=10, fontweight='bold')
    
    for h in range(heads):
        x = 1 + h * 2
        ax4.arrow(x+0.75, 5.5, (5-x-0.75)*0.9, 1.3,
                 head_width=0.12, head_length=0.08, fc='gray', ec='gray', alpha=0.6)
    
    linear = FancyBboxPatch((3, 8.5), 4, 0.8, boxstyle="round,pad=0.05",
                            facecolor='lightblue', edgecolor='black', lw=2)
    ax4.add_patch(linear)
    ax4.text(5, 8.9, 'Linear', ha='center', fontsize=10, fontweight='bold')
    
    output = FancyBboxPatch((3.5, 10), 3, 0.8, boxstyle="round,pad=0.05",
                            facecolor='lightcoral', edgecolor='black', lw=2)
    ax4.add_patch(output)
    ax4.text(5, 10.4, 'Output', ha='center', fontsize=10, fontweight='bold')
    
    ax4.arrow(5, 7.8, 0, 0.5, head_width=0.2, head_length=0.12, fc='black', ec='black', lw=2)
    ax4.arrow(5, 9.3, 0, 0.5, head_width=0.2, head_length=0.12, fc='black', ec='black', lw=2)
    
    # Positional encoding
    ax5 = fig.add_subplot(gs[2, 1:3])
    ax5.set_title('Positional Encoding Pattern', fontweight='bold', fontsize=13)
    
    max_len, d_model = 50, 128
    pos = np.arange(max_len)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    
    im = ax5.imshow(pe.T, cmap='RdBu_r', aspect='auto')
    ax5.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Dimension', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax5, label='Value', fraction=0.046)
    
    formula = "PE(pos,2i)=sin(pos/10000^(2i/d))\nPE(pos,2i+1)=cos(pos/10000^(2i/d))"
    ax5.text(0.98, 0.98, formula, transform=ax5.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.9), family='monospace')
    
    # Attention flow example
    ax6 = fig.add_subplot(gs[2:, 3])
    ax6.set_title('Attention Flow\nExample', fontweight='bold', fontsize=12)
    
    tokens = ['The', 'cat', 'sat', 'on', 'mat']
    query_idx = 2
    
    np.random.seed(42)
    weights = np.array([0.05, 0.3, 0.15, 0.35, 0.15])
    weights = weights / weights.sum()
    
    y_pos = np.arange(len(tokens))
    colors = plt.cm.viridis(weights / weights.max())
    
    ax6.barh(y_pos, weights, color=colors, edgecolor='black', lw=1.5)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(tokens, fontsize=11, fontweight='bold')
    ax6.set_xlabel('Attention Weight', fontsize=11, fontweight='bold')
    ax6.set_xlim(0, max(weights)*1.2)
    ax6.grid(True, alpha=0.3, axis='x')
    
    for i, (token, w) in enumerate(zip(tokens, weights)):
        ax6.text(w + 0.01, i, f'{w:.3f}', va='center', fontsize=10, fontweight='bold')
    
    ax6.axhline(query_idx, color='red', lw=3, alpha=0.5, label=f'Query: "{tokens[query_idx]}"')
    ax6.legend(loc='upper right', fontsize=10)
    
    # Complexity comparison
    ax7 = fig.add_subplot(gs[3, :2])
    ax7.set_title('Computational Complexity', fontweight='bold', fontsize=13)
    
    n = np.arange(10, 1001, 10)
    d = 512
    
    self_attn = n**2 * d
    recurrent = n * d**2
    conv = n * d**2 * 3  # kernel size k=3
    
    ax7.plot(n, self_attn, 'b-', lw=2.5, label='Self-Attention: O(n²·d)', alpha=0.8)
    ax7.plot(n, recurrent, 'r-', lw=2.5, label='Recurrent: O(n·d²)', alpha=0.8)
    ax7.plot(n, conv, 'g-', lw=2.5, label='Convolutional: O(k·n·d²)', alpha=0.8)
    
    ax7.fill_between(n, self_attn, alpha=0.2)
    ax7.grid(True, alpha=0.4)
    ax7.legend(fontsize=11)
    ax7.set_xlabel('Sequence Length n', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Operations', fontsize=12, fontweight='bold')
    ax7.set_yscale('log')
    
    # Parallelization
    ax8 = fig.add_subplot(gs[3, 2:])
    ax8.set_title('Parallelization Capability', fontweight='bold', fontsize=13)
    
    models = ['RNN/LSTM', 'CNN', 'Transformer']
    sequential = [100, 30, 5]
    parallel = [5, 90, 95]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax8.bar(x - width/2, sequential, width, label='Sequential', 
                    color='coral', alpha=0.8, edgecolor='black')
    bars2 = ax8.bar(x + width/2, parallel, width, label='Parallel',
                    color='lightblue', alpha=0.8, edgecolor='black')
    
    ax8.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax8.legend(fontsize=11)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.set_ylim(0, 110)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

#==============================================================================
# SECTION 5: IMAGE SEGMENTATION
#==============================================================================

def visualize_segmentation():
    """Segmentation: Pixel-wise classification"""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)
    fig.suptitle('IMAGE SEGMENTATION: Pixel-wise Dense Prediction', 
                 fontsize=18, fontweight='bold')
    
    # U-Net architecture
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 16)
    ax1.axis('off')
    ax1.set_title('U-Net Architecture', fontweight='bold', fontsize=14)
    
    # Encoder path (contracting)
    enc_levels = [
        (1, 1, 1.5, 1, '256×256', 'lightblue'),
        (1.5, 3, 1.5, 0.9, '128×128', 'lightgreen'),
        (2, 5, 1.5, 0.8, '64×64', 'lightyellow'),
        (2.5, 7, 1.5, 0.7, '32×32', 'lightcoral'),
        (3, 9, 1.5, 0.6, '16×16', 'plum'),
    ]
    
    for x, y, w, h, label, color in enc_levels:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                            facecolor=color, edgecolor='black', lw=2)
        ax1.add_patch(box)
        ax1.text(x+w/2, y+h/2, f'Conv\n{label}', ha='center', va='center',
                fontsize=8, fontweight='bold')
        
        if y < 9:
            ax1.arrow(x+w/2, y+h, 0, 0.8, head_width=0.15, head_length=0.1,
                     fc='blue', ec='blue', lw=2)
            ax1.text(x+w/2+0.5, y+h+0.4, 'MaxPool', fontsize=7, rotation=90)
    
    # Bottleneck
    bottleneck = FancyBboxPatch((3, 11), 1.5, 0.5, boxstyle="round,pad=0.08",
                                facecolor='orange', edgecolor='black', lw=2)
    ax1.add_patch(bottleneck)
    ax1.text(3.75, 11.25, 'Bottleneck\n8×8', ha='center', va='center',
            fontsize=8, fontweight='bold')
    
    ax1.arrow(3.75, 9.6, 0, 1.2, head_width=0.15, head_length=0.1,
             fc='blue', ec='blue', lw=2)
    
    # Decoder path (expanding)
    dec_levels = [
        (7, 9, 1.5, 0.6, '16×16', 'plum'),
        (7.5, 7, 1.5, 0.7, '32×32', 'lightcoral'),
        (8, 5, 1.5, 0.8, '64×64', 'lightyellow'),
        (8.5, 3, 1.5, 0.9, '128×128', 'lightgreen'),
        (9, 1, 1.5, 1, '256×256', 'lightblue'),
    ]
    
    for x, y, w, h, label, color in dec_levels:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                            facecolor=color, edgecolor='black', lw=2)
        ax1.add_patch(box)
        ax1.text(x+w/2, y+h/2, f'UpConv\n{label}', ha='center', va='center',
                fontsize=8, fontweight='bold')
        
        if y > 1:
            ax1.arrow(x+w/2, y, 0, -0.8, head_width=0.15, head_length=0.1,
                     fc='red', ec='red', lw=2)
            ax1.text(x+w/2-0.5, y-0.4, 'UpSample', fontsize=7, rotation=90)
    
    # Skip connections
    skip_pairs = [(1.75, 9.5, 7.75), (2.25, 7.5, 8.25), (2.75, 5.5, 8.75),
                  (3.25, 3.5, 9.25), (3.75, 1.5, 9.75)]
    
    for x_enc, y, x_dec in skip_pairs:
        ax1.annotate('', xy=(x_dec, y), xytext=(x_enc+1.5, y),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='green',
                                  connectionstyle='arc3,rad=0'))
        ax1.text((x_enc+x_dec+1.5)/2, y+0.2, 'concat', ha='center',
                fontsize=7, color='green', fontweight='bold')
    
    ax1.arrow(3.75, 11.5, 2.5, -2, head_width=0.15, head_length=0.1,
             fc='red', ec='red', lw=2)
    
    ax1.text(1, 0.2, 'ENCODER\n(Downsample)', ha='center', fontsize=11,
            fontweight='bold', color='blue',
            bbox=dict(boxstyle='round', fc='lightblue', alpha=0.9))
    ax1.text(9, 0.2, 'DECODER\n(Upsample)', ha='center', fontsize=11,
            fontweight='bold', color='red',
            bbox=dict(boxstyle='round', fc='lightcoral', alpha=0.9))
    
    # Segmentation example
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Input Image', fontweight='bold', fontsize=13)
    np.random.seed(42)
    img = np.random.rand(64, 64, 3) * 0.5 + 0.3
    # Add shapes
    img[15:25, 20:40] = [0.8, 0.2, 0.2]  # Red rectangle
    img[35:50, 30:45] = [0.2, 0.8, 0.2]  # Green rectangle
    ax2.imshow(img)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Ground Truth Segmentation', fontweight='bold', fontsize=13)
    seg_gt = np.zeros((64, 64))
    seg_gt[15:25, 20:40] = 1  # Class 1
    seg_gt[35:50, 30:45] = 2  # Class 2
    im = ax3.imshow(seg_gt, cmap='Set1', vmin=0, vmax=3)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, ticks=[0, 1, 2], label='Class')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Predicted Segmentation', fontweight='bold', fontsize=13)
    seg_pred = seg_gt + np.random.rand(64, 64) * 0.3
    seg_pred = np.clip(seg_pred, 0, 2.5)
    im = ax4.imshow(seg_pred, cmap='Set1', vmin=0, vmax=3)
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, ticks=[0, 1, 2], label='Class')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title('Overlay: Prediction on Image', fontweight='bold', fontsize=13)
    overlay = img.copy()
    mask = seg_pred > 0.5
    overlay[mask] = overlay[mask] * 0.5 + np.array([0.5, 0.5, 0]) * 0.5
    ax5.imshow(overlay)
    ax5.axis('off')
    
    # Loss functions
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.set_title('Segmentation Loss Functions', fontweight='bold', fontsize=13)
    
    x = np.linspace(0.01, 0.99, 100)
    ce_loss = -np.log(x)  # Cross-entropy
    dice_loss = 1 - 2*x / (x + 1)  # Dice loss
    focal_loss = -(1-x)**2 * np.log(x)  # Focal loss
    
    ax6.plot(x, ce_loss, 'b-', lw=2.5, label='Cross-Entropy', alpha=0.8)
    ax6.plot(x, dice_loss, 'r-', lw=2.5, label='Dice Loss', alpha=0.8)
    ax6.plot(x, focal_loss, 'g-', lw=2.5, label='Focal Loss', alpha=0.8)
    
    ax6.fill_between(x, ce_loss, alpha=0.15)
    ax6.grid(True, alpha=0.4)
    ax6.legend(fontsize=11)
    ax6.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, 5)
    
    # Metrics
    ax7 = fig.add_subplot(gs[3, :])
    ax7.set_title('Segmentation Metrics Over Training', fontweight='bold', fontsize=13)
    
    epochs = np.arange(1, 51)
    iou = 0.95 * (1 - np.exp(-epochs/10)) + 0.05*np.random.randn(len(epochs))*0.1
    dice = 0.96 * (1 - np.exp(-epochs/10)) + 0.04*np.random.randn(len(epochs))*0.1
    pixel_acc = 0.98 * (1 - np.exp(-epochs/8)) + 0.02*np.random.randn(len(epochs))*0.05
    
    ax7.plot(epochs, iou, 'b-', lw=2.5, label='IoU (Intersection over Union)', alpha=0.8)
    ax7.plot(epochs, dice, 'r-', lw=2.5, label='Dice Coefficient', alpha=0.8)
    ax7.plot(epochs, pixel_acc, 'g-', lw=2.5, label='Pixel Accuracy', alpha=0.8)
    
    ax7.fill_between(epochs, iou, alpha=0.15)
    ax7.grid(True, alpha=0.4)
    ax7.legend(fontsize=11, loc='lower right')
    ax7.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax7.set_ylim(0, 1.1)
    
    # Add formulas
    formulas = ("Metrics:\nIoU = |A∩B| / |A∪B|\nDice = 2|A∩B| / (|A|+|B|)\nPixel Acc = Correct / Total")
    ax7.text(0.98, 0.05, formulas, transform=ax7.transAxes,
            fontsize=9, va='bottom', ha='right', family='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.95))
    
    plt.tight_layout()
    return fig

#==============================================================================
# SECTION 6: EFFICIENT ARCHITECTURES - "LESS IS MORE"
#==============================================================================

def visualize_efficient_architectures():
    """Efficient models: MobileNet, EfficientNet, Vision Transformers"""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35)
    fig.suptitle('EFFICIENT ARCHITECTURES: Less is More Philosophy', 
                 fontsize=18, fontweight='bold')
    
    # Depthwise Separable Convolution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Depthwise Separable\nConvolution', fontweight='bold', fontsize=12)
    
    # Standard conv
    std_box = FancyBboxPatch((0.5, 8), 3, 2, boxstyle="round,pad=0.1",
                             facecolor='lightcoral', edgecolor='black', lw=2)
    ax1.add_patch(std_box)
    ax1.text(2, 9, 'Standard Conv\n3×3×C filters\nParams: 9×C²', ha='center', va='center',
            fontsize=9, fontweight='bold')
    
    # Depthwise
    dw_box = FancyBboxPatch((5, 9), 2, 1.3, boxstyle="round,pad=0.1",
                            facecolor='lightblue', edgecolor='black', lw=2)
    ax1.add_patch(dw_box)
    ax1.text(6, 9.65, 'Depthwise\n3×3 per channel\nParams: 9×C', ha='center', va='center',
            fontsize=8, fontweight='bold')
    
    # Pointwise
    pw_box = FancyBboxPatch((5, 7), 2, 1.3, boxstyle="round,pad=0.1",
                            facecolor='lightgreen', edgecolor='black', lw=2)
    ax1.add_patch(pw_box)
    ax1.text(6, 7.65, 'Pointwise\n1×1×C\nParams: C²', ha='center', va='center',
            fontsize=8, fontweight='bold')
    
    ax1.arrow(6, 8.4, 0, 0.4, head_width=0.2, head_length=0.12, fc='k', ec='k', lw=2)
    
    ax1.text(6, 6, f'Reduction:\n9C² → (9C + C²)\n≈9× fewer params\nwhen C is large',
            ha='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', fc='yellow', alpha=0.9))
    
    # Inverted Residual Block (MobileNetV2)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Inverted Residual\n(MobileNetV2)', fontweight='bold', fontsize=12)
    
    ax2.text(5, 11, 'Input\nC channels', ha='center', fontsize=10, fontweight='bold')
    
    expand = FancyBboxPatch((3.5, 9), 3, 0.8, boxstyle="round,pad=0.08",
                           facecolor='lightgreen', edgecolor='black', lw=2)
    ax2.add_patch(expand)
    ax2.text(5, 9.4, 'Expand 1×1\n→ t×C channels', ha='center', fontsize=8, fontweight='bold')
    
    dw = FancyBboxPatch((3.5, 7.5), 3, 0.8, boxstyle="round,pad=0.08",
                        facecolor='lightblue', edgecolor='black', lw=2)
    ax2.add_patch(dw)
    ax2.text(5, 7.9, 'Depthwise 3×3', ha='center', fontsize=8, fontweight='bold')
    
    project = FancyBboxPatch((3.5, 6), 3, 0.8, boxstyle="round,pad=0.08",
                            facecolor='lightcoral', edgecolor='black', lw=2)
    ax2.add_patch(project)
    ax2.text(5, 6.4, 'Project 1×1\n→ C channels', ha='center', fontsize=8, fontweight='bold')
    
    for y in [8.8, 7.3]:
        ax2.arrow(5, y, 0, -0.5, head_width=0.2, head_length=0.1, fc='k', ec='k', lw=2)
    
    # Skip connection
    ax2.annotate('', xy=(7.5, 5.5), xytext=(7.5, 11),
                arrowprops=dict(arrowstyle='->', lw=3, color='red',
                              connectionstyle='arc3,rad=0.5'))
    ax2.text(8.5, 8, 'Skip', ha='center', fontsize=10, fontweight='bold', color='red')
    
    add = Circle((5, 4.5), 0.4, fc='yellow', ec='black', lw=2)
    ax2.add_patch(add)
    ax2.text(5, 4.5, '+', ha='center', va='center', fontsize=18, fontweight='bold')
    
    ax2.arrow(5, 5.8, 0, -1, head_width=0.2, head_length=0.1, fc='k', ec='k', lw=2)
    ax2.text(5, 3.5, 'Output', ha='center', fontsize=10, fontweight='bold')
    
    # Squeeze-and-Excitation
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 12)
    ax3.axis('off')
    ax3.set_title('Squeeze-Excitation\nAttention', fontweight='bold', fontsize=12)
    
    ax3.text(5, 11, 'Features\nH×W×C', ha='center', fontsize=10, fontweight='bold')
    
    squeeze = FancyBboxPatch((3.5, 9), 3, 0.8, boxstyle="round,pad=0.08",
                            facecolor='lightblue', edgecolor='black', lw=2)
    ax3.add_patch(squeeze)
    ax3.text(5, 9.4, 'Global Pool\n→ 1×1×C', ha='center', fontsize=8, fontweight='bold')
    
    fc1 = FancyBboxPatch((3.5, 7.5), 3, 0.8, boxstyle="round,pad=0.08",
                        facecolor='lightgreen', edgecolor='black', lw=2)
    ax3.add_patch(fc1)
    ax3.text(5, 7.9, 'FC + ReLU\n→ C/r', ha='center', fontsize=8, fontweight='bold')
    
    fc2 = FancyBboxPatch((3.5, 6), 3, 0.8, boxstyle="round,pad=0.08",
                        facecolor='lightcoral', edgecolor='black', lw=2)
    ax3.add_patch(fc2)
    ax3.text(5, 6.4, 'FC + Sigmoid\n→ C', ha='center', fontsize=8, fontweight='bold')
    
    for y in [8.8, 7.3, 5.8]:
        if y == 5.8:
            ax3.arrow(5, y, 0, -0.3, head_width=0.2, head_length=0.1, fc='k', ec='k', lw=2)
        else:
            ax3.arrow(5, y, 0, -0.5, head_width=0.2, head_length=0.1, fc='k', ec='k', lw=2)
    
    scale = FancyBboxPatch((3.5, 4), 3, 0.8, boxstyle="round,pad=0.08",
                          facecolor='yellow', edgecolor='black', lw=2)
    ax3.add_patch(scale)
    ax3.text(5, 4.4, 'Scale Features\n× weights', ha='center', fontsize=8, fontweight='bold')
    
    ax3.text(5, 3, 'Recalibrated\nOutput', ha='center', fontsize=10, fontweight='bold')
    
    # Efficiency comparison
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title('Model Efficiency: Accuracy vs Parameters', fontweight='bold', fontsize=14)
    
    models = {
        'VGG16': (138, 71.5),
        'ResNet50': (25.6, 76.1),
        'ResNet101': (44.5, 77.4),
        'MobileNetV2': (3.5, 72.0),
        'EfficientNet-B0': (5.3, 77.1),
        'EfficientNet-B4': (19, 82.9),
        'ViT-Base': (86, 81.8),
        'DeiT-Tiny': (5.7, 72.2),
    }
    
    params = [v[0] for v in models.values()]
    accs = [v[1] for v in models.values()]
    labels = list(models.keys())
    
    colors = ['red', 'blue', 'blue', 'green', 'orange', 'orange', 'purple', 'purple']
    
    for i, (p, a, l, c) in enumerate(zip(params, accs, labels, colors)):
        ax4.scatter(p, a, s=300, alpha=0.7, c=c, edgecolors='black', lw=2)
        ax4.annotate(l, (p, a), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    ax4.grid(True, alpha=0.4)
    ax4.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('ImageNet Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_xlim(1, 200)
    ax4.set_ylim(70, 85)
    
    # Add efficiency frontier
    efficient_x = [3.5, 5.3, 19]
    efficient_y = [72.0, 77.1, 82.9]
    ax4.plot(efficient_x, efficient_y, 'orange', linestyle='--', lw=2,
            label='Efficiency Frontier', alpha=0.7)
    ax4.legend(fontsize=11)
    
    # FLOPs comparison
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_title('Computational Cost (FLOPs) Comparison', fontweight='bold', fontsize=13)
    
    model_names = ['VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNet-B0']
    flops = [15.5, 4.1, 0.3, 0.39]
    
    bars = ax5.barh(model_names, flops, color=['red', 'blue', 'green', 'orange'],
                    alpha=0.7, edgecolor='black', lw=2)
    
    for i, (bar, flop) in enumerate(zip(bars, flops)):
        ax5.text(flop + 0.5, i, f'{flop:.2f}B', va='center',
                fontsize=10, fontweight='bold')
    
    ax5.set_xlabel('FLOPs (Billions)', fontsize=12, fontweight='bold')
    ax5.set_xlim(0, 18)
    ax5.grid(True, alpha=0.4, axis='x')
    
    # Compound scaling
    ax6 = fig.add_subplot(gs[2, 2], projection='3d')
    ax6.set_title('EfficientNet\nCompound Scaling', fontweight='bold', fontsize=12)
    
    depth = np.array([1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1, 3.6])
    width = np.array([1, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
    resolution = np.array([224, 240, 260, 300, 380, 456, 528, 600])
    accuracy = np.array([77.1, 79.1, 80.1, 81.6, 82.9, 83.6, 84.3, 84.3])
    
    colors_3d = plt.cm.viridis((accuracy - accuracy.min()) / (accuracy.max() - accuracy.min()))
    
    ax6.scatter(depth, width, resolution, c=colors_3d, s=200, 
               edgecolors='black', lw=2, alpha=0.8)
    
    for i in range(len(depth)):
        ax6.text(depth[i], width[i], resolution[i], f'B{i}',
                fontsize=8, fontweight='bold')
    
    ax6.set_xlabel('Depth', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Width', fontsize=10, fontweight='bold')
    ax6.set_zlabel('Resolution', fontsize=10, fontweight='bold')
    ax6.view_init(elev=20, azim=45)
    
    # NAS and AutoML
    ax7 = fig.add_subplot(gs[3, :])
    ax7.set_title('Neural Architecture Search: Evolution of Efficiency', fontweight='bold', fontsize=14)
    
    generations = np.arange(1, 51)
    
    baseline = np.ones(len(generations)) * 75
    manual = 75 + 2 * (1 - np.exp(-generations/15)) + 0.5*np.random.randn(len(generations))
    nas = 75 + 5 * (1 - np.exp(-generations/10)) + 0.3*np.random.randn(len(generations))
    efficient_nas = 75 + 7 * (1 - np.exp(-generations/8)) + 0.2*np.random.randn(len(generations))
    
    ax7.plot(generations, baseline, 'k--', lw=2, label='Baseline (Manual)', alpha=0.7)
    ax7.plot(generations, manual, 'b-', lw=2, label='Manual Design', alpha=0.8)
    ax7.plot(generations, nas, 'g-', lw=2.5, label='NAS (Basic)', alpha=0.8)
    ax7.plot(generations, efficient_nas, 'orange', lw=2.5, 
            label='Efficient NAS (MobileNet, EfficientNet)', alpha=0.8)
    
    ax7.fill_between(generations, baseline, manual, alpha=0.15, color='blue')
    ax7.fill_between(generations, manual, nas, alpha=0.15, color='green')
    ax7.fill_between(generations, nas, efficient_nas, alpha=0.15, color='orange')
    
    ax7.grid(True, alpha=0.4)
    ax7.legend(fontsize=11, loc='lower right')
    ax7.set_xlabel('Search Iterations / Design Cycles', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Model Performance (%)', fontsize=12, fontweight='bold')
    ax7.set_ylim(74, 84)
    
    # Key insights
    insights = ("Key Principles:\n"
               "• Depthwise separable convolutions\n"
               "• Inverted residuals\n"
               "• Squeeze-Excitation attention\n"
               "• Compound scaling (d, w, r)\n"
               "• Neural Architecture Search\n"
               "Result: 10-100× efficiency gain")
    ax7.text(0.98, 0.05, insights, transform=ax7.transAxes,
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.95,
                     edgecolor='orange', lw=2))
    
    plt.tight_layout()
    return fig

#==============================================================================
# MAIN EXECUTION FUNCTION
#==============================================================================

def visualize_all_architectures():
    """
    Execute all visualizations sequentially.
    This creates a complete visual journey through AI evolution.
    """
    print("\n" + "="*80)
    print("GENERATING COMPLETE AI ARCHITECTURE VISUALIZATIONS")
    print("="*80 + "\n")
    
    sections = [
        ("1. Convolution Fundamentals", visualize_convolution_math),
        ("2. LSTM Architecture", visualize_lstm_complete),
        ("3. ResNet Residual Learning", visualize_resnet_complete),
        ("4. Transformer Attention", visualize_transformer_complete),
        ("5. Image Segmentation", visualize_segmentation),
        ("6. Efficient Architectures", visualize_efficient_architectures),
    ]
    
    figures = []
    for name, func in sections:
        print(f"Generating {name}...")
        try:
            fig = func()
            figures.append((name, fig))
            print(f"✓ {name} complete\n")
        except Exception as e:
            print(f"✗ Error in {name}: {str(e)}\n")
    
    print("="*80)
    print(f"GENERATED {len(figures)} VISUALIZATION SECTIONS")
    print("="*80)
    print("\nAll figures are now displayed.")
    print("Use plt.show() to view them interactively.")
    print("\nTo save figures:")
    print("  for name, fig in figures:")
    print("    fig.savefig(f'{name}.png', dpi=300, bbox_inches='tight')")
    
    plt.show()
    return figures

#==============================================================================
# MATHEMATICAL SUMMARY
#==============================================================================

def print_mathematical_summary():
    """Print comprehensive mathematical summary of all architectures"""
    summary = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MATHEMATICAL ARCHITECTURE SUMMARY                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. CONVOLUTION
   Operation: Y[i,j] = Σ_m Σ_n X[i+m,j+n] · K[m,n] + b
   Properties: Translation equivariance, local connectivity
   Parameters: K²·C_in·C_out (K=kernel size, C=channels)

2. LSTM (Long Short-Term Memory)
   Forget Gate:    f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
   Input Gate:     i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
   Candidate:      C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)
   Cell Update:    C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
   Output Gate:    o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
   Hidden State:   h_t = o_t ⊙ tanh(C_t)
   
   Key: Gating mechanism enables long-term dependencies

3. RESNET (Residual Networks)
   Core Equation: y = F(x, {W_i}) + x
   
   Where:
   - F(x): Residual function (learned transformation)
   - x: Identity mapping (skip connection)
   
   Innovation: Gradient flow through skip connections
   Enables: Training 100+ layer networks

4. TRANSFORMER
   Self-Attention: Attention(Q, K, V) = softmax(QK^T/√d_k)·V
   
   Where:
   - Q = X·W_Q (Query matrix)
   - K = X·W_K (Key matrix)
   - V = X·W_V (Value matrix)
   - d_k: Dimension of keys (for scaling)
   
   Multi-Head: MultiHead(Q,K,V) = Concat(head_1,...,head_h)·W_O
               head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
   
   Positional Encoding:
   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   
   Complexity: O(n²·d) per layer (n=sequence length)

5. IMAGE SEGMENTATION
   Task: Pixel-wise classification
   U-Net: Encoder-Decoder with skip connections
   
   Loss Functions:
   - Cross-Entropy: L = -Σ y_i·log(ŷ_i)
   - Dice Loss: L = 1 - 2·|Y∩Ŷ| / (|Y|+|Ŷ|)
   - IoU: |Y∩Ŷ| / |Y∪Ŷ|

6. EFFICIENT ARCHITECTURES
   Depthwise Separable Convolution:
   - Depthwise: 3×3 per channel → 9C parameters
   - Pointwise: 1×1 across channels → C² parameters
   - Total: 9C + C² vs Standard 9C²
   - Reduction: ~9× when C is large
   
   Inverted Residual:
   - Expand: 1×1 (C → tC)
   - Depthwise: 3×3 (tC → tC)
   - Project: 1×1 (tC → C)
   
   Compound Scaling (EfficientNet):
   - depth: d = α^φ
   - width: w = β^φ
   - resolution: r = γ^φ
   - Constraint: α·β²·γ² ≈ 2

╔══════════════════════════════════════════════════════════════════════════════╗
║                         EVOLUTION TIMELINE                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

1997: LSTM - Hochreiter & Schmidhuber
2012: AlexNet - Krizhevsky et al. (Deep Learning Revolution)
2014: VGGNet, GoogLeNet/Inception
2015: ResNet - He et al. (Residual Learning)
2015: U-Net - Ronneberger et al. (Segmentation)
2017: Transformer - Vaswani et al. (Attention Is All You Need)
2017: MobileNet - Howard et al. (Efficient CNNs)
2019: EfficientNet - Tan & Le (Compound Scaling)
2020: Vision Transformer (ViT) - Dosovitskiy et al.
2021-2025: Efficient Transformers, Neural Architecture Search

╔══════════════════════════════════════════════════════════════════════════════╗
║                       KEY INSIGHTS FOR STUDENTS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. LOCAL → GLOBAL: CNNs extract local features, Transformers capture global context
2. MEMORY: LSTMs gate information flow, Transformers use attention
3. DEPTH: ResNets enable deep networks via residual connections
4. EFFICIENCY: Modern architectures achieve more with less computation
5. ARCHITECTURE SEARCH: AutoML discovers optimal designs
6. TRADE-OFFS: Accuracy vs Speed vs Parameters vs FLOPs

The future: Hybrid architectures combining strengths of each paradigm
"""
    print(summary)

# Print summary on load
print_mathematical_summary()

print("\n" + "="*80)
print("READY TO VISUALIZE!")
print("="*80)
print("\nExecute: visualize_all_architectures()")
print("         - Generates all 6 visualization sections")
print("\nOr run individual sections:")
print("  fig1 = visualize_convolution_math()")
print("  fig2 = visualize_lstm_complete()")
print("  fig3 = visualize_resnet_complete()")
print("  fig4 = visualize_transformer_complete()")
print("  fig5 = visualize_segmentation()")
print("  fig6 = visualize_efficient_architectures()")
print("\n" + "="*80)
