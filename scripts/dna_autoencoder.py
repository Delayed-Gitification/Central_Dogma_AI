import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class MonotonicExpansion(nn.Module):
    """
    Differentiable monotonic expansion from latent columns to output sequence.
    Uses soft assignment based on cumulative predicted emission masses.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z, mass, target_len):
        """
        z: (B, M, D) Latent columns
        mass: (B, M) Predicted emission masses (must be >= 0)
        target_len: int (L) Target sequence length
        """
        B, M, D = z.size()
        
        # Prepend 0 for C_{-1} equivalent
        mass_padded = torch.cat([torch.zeros(B, 1, device=z.device), mass], dim=1) # (B, M+1)
        C = torch.cumsum(mass_padded, dim=1) # (B, M+1)
        
        # C_prev and C_curr for the boundaries of each latent column
        C_prev = C[:, :-1].unsqueeze(1) # (B, 1, M)
        C_curr = C[:, 1:].unsqueeze(1)  # (B, 1, M)
        
        # Coordinates of the output sequence
        coords = torch.arange(target_len, device=z.device).float() + 0.5 # (L)
        coords = coords.unsqueeze(0).unsqueeze(-1) # (1, L, 1)
        
        # Compute soft attention weight
        # weight is high if coord falls between C_prev and C_curr
        w1 = torch.sigmoid((C_curr - coords) / self.temperature)
        w2 = torch.sigmoid((C_prev - coords) / self.temperature)
        weights = w1 - w2 # (B, L, M)
        
        # Normalize just to be strictly stable
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Expand latents
        # E_i = \sum_j W_{i, j} * Z_j
        E = torch.bmm(weights, z) # (B, L, D)
        
        return E, weights

class DNAEncoder(nn.Module):
    def __init__(self, latent_dim, target_bases_per_latent):
        super().__init__()
        self.stride = target_bases_per_latent
        self.conv = nn.Sequential(
            nn.Conv1d(4, latent_dim, kernel_size=self.stride*2, stride=self.stride, padding=self.stride//2),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1)
        )
        self.content_head = nn.Linear(latent_dim, latent_dim)
        # Softplus ensures emission mass is >= 0
        self.mass_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        h = self.conv(x) # (B, D, M)
        h = h.transpose(1, 2) # (B, M, D)
        
        content = self.content_head(h)
        mass = self.mass_head(h).squeeze(-1) # (B, M)
        return content, mass

class DNADecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Lightweight pointwise MLP because all the structural alignment is done in Expansion
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 4)
        )
        
    def forward(self, x):
        return self.net(x)

class DNAAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, target_bases_per_latent=20):
        super().__init__()
        self.encoder = DNAEncoder(latent_dim, target_bases_per_latent)
        self.expansion = MonotonicExpansion(temperature=0.1)
        self.decoder = DNADecoder(latent_dim)
        
    def forward(self, x):
        """
        x: (B, 4, L) One-hot DNA sequence
        """
        B, _, L = x.size()
        z, mass = self.encoder(x)
        E, weights = self.expansion(z, mass, L)
        logits = self.decoder(E) # (B, L, 4)
        return logits, mass, weights

def test_model(args):
    device = torch.device('mps' if args.mps and torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    model = DNAAutoencoder(latent_dim=args.latent_dim, target_bases_per_latent=args.target_bases_per_latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    B = 8
    L = args.max_seq_len
    
    print("\nStarting Training Dummy Loop (Differentiable Monotonic Expansion)...")
    for epoch in range(50):
        # Generate random high-entropy DNA sequences
        seq_indices = torch.randint(0, 4, (B, L), device=device)
        x = F.one_hot(seq_indices, num_classes=4).float().transpose(1, 2)
        
        optimizer.zero_grad()
        logits, mass, weights = model(x)
        
        # Reconstruction CE
        loss_ce = F.cross_entropy(logits.view(-1, 4), seq_indices.view(-1))
        
        # Length/Mass loss
        total_mass = mass.sum(dim=-1) # (B,)
        loss_length = F.mse_loss(total_mass, torch.full_like(total_mass, L))
        
        # Total Loss
        loss = loss_ce + 0.1 * loss_length
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == seq_indices).float().mean().item()
                exact_match = (preds == seq_indices).all(dim=-1).float().mean().item()
                
                print(f"Epoch {epoch+1:03d} | CE Loss: {loss_ce.item():.4f} | Len Loss: {loss_length.item():.4f}")
                print(f"  Diagnostics:")
                print(f"  - Hard-decoded accuracy: {acc:.4f}")
                print(f"  - Exact sequence reconstruction rate: {exact_match:.4f}")
                print(f"  - Predicted total length vs true length: {total_mass.mean().item():.1f} vs {L}")
                print(f"  - Emission mass distribution: mean={mass.mean().item():.2f}, min={mass.min().item():.2f}, max={mass.max().item():.2f}")
                zero_emit = (mass < 0.5).sum().item() / B
                print(f"  - Avg zero-emitting columns per sequence: {zero_emit:.1f}")
                print(f"  - Avg bases emitted per latent column: {total_mass.mean().item() / mass.size(1):.2f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Variable-emission latent DNA autoencoder")
    parser.add_argument('--target-bases-per-latent', type=int, default=20, help="Average bases encoded per latent column")
    parser.add_argument('--latent-dim', type=int, default=64, help="Dimensionality of the latent space")
    parser.add_argument('--max-seq-len', type=int, default=200, help="Target sequence length for dummy data")
    parser.add_argument('--mps', action='store_true', default=True, help="Use MPS if available")
    args = parser.parse_args()
    test_model(args)
