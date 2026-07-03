import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import difflib
import os

class MonotonicExpansion(nn.Module):
    """
    Differentiable monotonic expansion from latent columns to output sequence.
    Provides expanded latent representations AND positional features (absolute and relative).
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z, mass, target_len):
        B, M, D = z.size()
        
        # Prepend 0 for C_{-1} equivalent
        mass_padded = torch.cat([torch.zeros(B, 1, device=z.device), mass], dim=1) # (B, M+1)
        C = torch.cumsum(mass_padded, dim=1) # (B, M+1)
        
        C_prev = C[:, :-1].unsqueeze(1) # (B, 1, M)
        C_curr = C[:, 1:].unsqueeze(1)  # (B, 1, M)
        
        coords = torch.arange(target_len, device=z.device).float() + 0.5 # (L)
        coords = coords.unsqueeze(0).unsqueeze(-1) # (1, L, 1)
        
        w1 = torch.sigmoid((C_curr - coords) / self.temperature)
        w2 = torch.sigmoid((C_prev - coords) / self.temperature)
        weights = w1 - w2 # (B, L, M)
        
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        E = torch.bmm(weights, z) # (B, L, D)
        
        expected_C_prev = (weights * C_prev).sum(dim=-1, keepdim=True) # (B, L, 1)
        expected_mass = (weights * mass.unsqueeze(1)).sum(dim=-1, keepdim=True) # (B, L, 1)
        
        rel_pos = (coords - expected_C_prev) / (expected_mass + 1e-4) # (B, L, 1)
        abs_pos = coords / target_len # (1, L, 1)
        abs_pos = abs_pos.expand(B, -1, -1) # (B, L, 1)
        
        pos_features = torch.cat([abs_pos, rel_pos, expected_mass], dim=-1) # (B, L, 3)
        
        return E, weights, pos_features

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
        
        self.mass_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softplus()
        )
        # Initialize mass head bias so that softplus(bias) approx target_bases_per_latent
        # For x > 20, softplus(x) approx x
        nn.init.zeros_(self.mass_head[0].weight)
        nn.init.constant_(self.mass_head[0].bias, float(target_bases_per_latent))
        
    def forward(self, x):
        h = self.conv(x) # (B, D, M)
        h = h.transpose(1, 2) # (B, M, D)
        
        content = self.content_head(h)
        mass = self.mass_head(h).squeeze(-1) # (B, M)
        return content, mass

class ResidualBlock1D(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class DNADecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        in_dim = latent_dim + 3
        self.in_proj = nn.Conv1d(in_dim, latent_dim, 1)
        self.blocks = nn.Sequential(
            ResidualBlock1D(latent_dim, kernel_size=9),
            ResidualBlock1D(latent_dim, kernel_size=9),
            ResidualBlock1D(latent_dim, kernel_size=9)
        )
        self.out_proj = nn.Conv1d(latent_dim, 4, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2) # (B, in_dim, L)
        x = self.in_proj(x)
        x = self.blocks(x)
        logits = self.out_proj(x) # (B, 4, L)
        return logits.transpose(1, 2) # (B, L, 4)

class DNAAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, target_bases_per_latent=20, temperature=0.1):
        super().__init__()
        self.encoder = DNAEncoder(latent_dim, target_bases_per_latent)
        self.expansion = MonotonicExpansion(temperature=temperature)
        self.decoder = DNADecoder(latent_dim)
        
    def forward(self, x, force_target_len=None):
        B, _, L = x.size()
        z, mass = self.encoder(x)
        
        if force_target_len is not None:
            target_len = force_target_len
        elif self.training:
            target_len = L
        else:
            if B > 1:
                raise ValueError("Dynamic target length without force_target_len is only supported for batch size 1.")
            target_len = max(1, int(mass.sum().item()))
            
        E, weights, pos_features = self.expansion(z, mass, target_len)
        
        decoder_in = torch.cat([E, pos_features], dim=-1)
        logits = self.decoder(decoder_in) # (B, target_len, 4)
        
        return logits, mass, weights

def generate_synthetic_data(B, L, device):
    """ Generates sequences with strong mixtures of high and low entropy regions. """
    seq_indices = torch.randint(0, 4, (B, L), device=device)
    
    for b in range(B):
        num_blocks = torch.randint(2, 6, (1,)).item()
        for _ in range(num_blocks):
            tract_len = torch.randint(15, 60, (1,)).item()
            if tract_len > L: tract_len = L
            start = torch.randint(0, L - tract_len + 1, (1,)).item()
            
            block_type = torch.randint(0, 3, (1,)).item()
            if block_type == 0:
                base = torch.randint(0, 4, (1,)).item()
                seq_indices[b, start:start+tract_len] = base
            elif block_type == 1:
                base1 = torch.randint(0, 4, (1,)).item()
                base2 = torch.randint(0, 4, (1,)).item()
                motif = torch.tensor([base1, base2] * (tract_len // 2 + 1), device=device)[:tract_len]
                seq_indices[b, start:start+tract_len] = motif
            else:
                pyrimidines = torch.tensor([1, 3], device=device)
                choices = torch.randint(0, 2, (tract_len,), device=device)
                seq_indices[b, start:start+tract_len] = pyrimidines[choices]
                
    return seq_indices

def test_perturbation(model, args, device):
    print("\n" + "="*60)
    print("Latent Editability / Perturbation Test")
    print("="*60)
    
    seq_indices = generate_synthetic_data(1, args.max_seq_len, device)
    x = F.one_hot(seq_indices, num_classes=4).float().transpose(1, 2)
    
    model.eval()
    with torch.no_grad():
        logits_orig, mass_orig, _ = model(x, force_target_len=args.max_seq_len)
        preds_orig = logits_orig.argmax(dim=-1)
        orig_len = args.max_seq_len
        
        z, mass = model.encoder(x)
        M = mass.size(1)
        col_to_edit = M // 2
        
        mass_perturbed = mass.clone()
        mass_perturbed[0, col_to_edit] += 20.0
        new_len = int(mass_perturbed.sum().item())
        
        E_pert, _, pos_pert = model.expansion(z, mass_perturbed, target_len=new_len)
        decoder_in = torch.cat([E_pert, pos_pert], dim=-1)
        logits_pert = model.decoder(decoder_in)
        preds_pert = logits_pert.argmax(dim=-1)
        
    print(f"Original predicted sequence length: {orig_len} bases")
    print(f"Added +20 emission mass to latent column index {col_to_edit} (out of {M}).")
    print(f"Perturbed dynamic sequence length: {new_len} bases")
    
    orig_seq = "".join(['ACGT'[i] for i in preds_orig[0]])
    pert_seq = "".join(['ACGT'[i] for i in preds_pert[0]])
    
    print(f"Original generated sequence length: {len(orig_seq)}")
    print(f"Perturbed generated sequence length: {len(pert_seq)}")
    
    # Compute alignment using difflib
    matcher = difflib.SequenceMatcher(None, orig_seq, pert_seq)
    blocks = matcher.get_opcodes()
    
    insertions_found = 0
    print("\nAlignment differences:")
    for tag, i1, i2, j1, j2 in blocks:
        if tag != 'equal':
            print(f"  {tag:7s} orig[{i1:3d}:{i2:3d}] -> pert[{j1:3d}:{j2:3d}]")
            if tag in ('insert', 'replace'):
                print(f"    orig segment: {orig_seq[i1:i2]}")
                print(f"    pert segment: {pert_seq[j1:j2]}")
                insertions_found += (j2 - j1) - (i2 - i1)
    
    if insertions_found > 0:
        print(f"\nSuccess: Found net insertions matching the expected +20 mass edit.")
    else:
        print("\nWarning: sequence did not cleanly expand locally as expected.")
    print("="*60 + "\n")

def train_dummy_loop(args):
    device = torch.device('mps' if args.mps and torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    model = DNAAutoencoder(
        latent_dim=args.latent_dim, 
        target_bases_per_latent=args.target_bases_per_latent,
        temperature=args.temperature
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    B = args.batch_size
    L = args.max_seq_len
    
    # Held-out validation batch
    val_seq = generate_synthetic_data(args.batch_size, L, device)
    val_x = F.one_hot(val_seq, num_classes=4).float().transpose(1, 2)
    
    print(f"\nStarting Training (V3 Architecture) for {args.epochs} steps...")
    for epoch in range(args.epochs):
        model.train()
        seq_indices = generate_synthetic_data(B, L, device)
        x = F.one_hot(seq_indices, num_classes=4).float().transpose(1, 2)
        
        optimizer.zero_grad()
        logits, mass, weights = model(x)
        
        # Reconstruction CE
        loss_ce = F.cross_entropy(logits.reshape(-1, 4), seq_indices.reshape(-1))
        
        # Length matching loss
        total_mass = mass.sum(dim=-1) # (B,)
        loss_length = F.mse_loss(total_mass, torch.full_like(total_mass, L))
        
        # Mass spread regularizer is now removed from the optimization objective!
        # It encourages uniform emission, which fights our manifold design.
        
        loss = loss_ce + 0.1 * loss_length
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % args.print_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # Training batch stats
                preds = logits.argmax(dim=-1)
                acc = (preds == seq_indices).float().mean().item()
                exact_match = (preds == seq_indices).all(dim=-1).float().mean().item()
                loss_variance = mass.var(dim=-1).mean().item()
                
                # Validation batch stats
                val_logits, val_mass, _ = model(val_x, force_target_len=L)
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == val_seq).float().mean().item()
                val_exact = (val_preds == val_seq).all(dim=-1).float().mean().item()
                val_ce = F.cross_entropy(val_logits.reshape(-1, 4), val_seq.reshape(-1)).item()
                
                print(f"Step {epoch+1:05d}/{args.epochs} | CE: {loss_ce.item():.4f} | Len: {loss_length.item():.4f} | Val CE: {val_ce:.4f}")
                print(f"  Diagnostics:")
                print(f"  - Train Acc: {acc:.4f} (Exact: {exact_match:.4f}) | Val Acc: {val_acc:.4f} (Exact: {val_exact:.4f})")
                print(f"  - Predicted length vs true length: {total_mass.mean().item():.1f} vs {L}")
                print(f"  - Mass stats: mean={mass.mean().item():.2f}, min={mass.min().item():.2f}, max={mass.max().item():.2f}, var={loss_variance:.2f}")
                zero_emit = (mass < 0.5).sum().item() / B
                print(f"  - Avg zero-emitting columns per sequence: {zero_emit:.1f}")
            
            # Save checkpoint
            if getattr(args, 'checkpoint_path', None):
                os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_path)), exist_ok=True)
                torch.save(model.state_dict(), args.checkpoint_path)

    # Run perturbation diagnostics after full training
    test_perturbation(model, args, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Variable-emission latent DNA autoencoder")
    parser.add_argument('--target-bases-per-latent', type=int, default=20)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--max-seq-len', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.1, help="Softmax/Sigmoid temperature for boundary expansion")
    parser.add_argument('--epochs', type=int, default=10000, help="Number of training steps")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size per step")
    parser.add_argument('--print-every', type=int, default=500, help="Print diagnostics every N steps")
    parser.add_argument('--checkpoint-path', type=str, default="checkpoints/dna_autoencoder.pt", help="Path to save the model checkpoint")
    parser.add_argument('--mps', action='store_true', default=True)
    args = parser.parse_args()
    train_dummy_loop(args)
