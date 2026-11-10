// GariKhata numerics: a section (interval) ledger for one private bus.
// Style: records, IReadOnlyList / ImmutableArray, static functions that are pure in their
// arguments (every stochastic function takes an explicit seed; mutation is local only).
using System.Collections.Immutable;
using System.Globalization;
using System.Numerics;
using System.Text.Json;

namespace GariKhata;

public sealed record Stop(string Name, double KmFromOrigin);
public sealed record Request(int From, int To);
public enum Pay { Upi, Cash }
public sealed record Pax(int Seat, int From, int To, int Fare, Pay Mode, bool Shows, bool Resold, bool Declared);
public sealed record Table(string Id, string Title, IReadOnlyList<string> Headers, IReadOnlyList<IReadOnlyList<string>> Rows);

public static class Route
{
    // ILLUSTRATIVE stop list and distances; to be replaced by the stand's own fare chart.
    public static readonly IReadOnlyList<Stop> Stops = new[]
    {
        new Stop("Bishnupur", 0), new Stop("Joypur", 14), new Stop("Kotulpur", 30),
        new Stop("Arambagh", 54), new Stop("Champadanga", 82), new Stop("Dankuni", 120),
        new Stop("Esplanade", 140),
    };
    public const int Seats = 40;
    public const int FullFare = 240;
    public static int Sections => Stops.Count - 1;
    public static int FullMask => (1 << Sections) - 1;
    public static int Mask(int from, int to) => ((1 << to) - 1) & ~((1 << from) - 1);
    public static double Km(int from, int to) => Stops[to].KmFromOrigin - Stops[from].KmFromOrigin;
    public static int Fare(int from, int to) =>
        Math.Max(20, 5 * (int)Math.Round(FullFare * Km(from, to) / Km(0, Sections) / 5.0));
}

/// Immutable seat-by-section ledger: one bitmask of occupied sections per seat.
public sealed record Ledger(ImmutableArray<int> SeatMasks)
{
    public static Ledger Empty => new(ImmutableArray.CreateRange(Enumerable.Repeat(0, Route.Seats)));
    public Ledger Place(int seat, int mask) => new(SeatMasks.SetItem(seat, SeatMasks[seat] | mask));
    public Ledger Free(int seat, int mask) => new(SeatMasks.SetItem(seat, SeatMasks[seat] & ~mask));
    public int Load(int section) => SeatMasks.Count(m => (m & (1 << section)) != 0);
    public bool LoadFeasible(int mask) =>
        Enumerable.Range(0, Route.Sections).Where(s => (mask & (1 << s)) != 0).All(s => Load(s) < Route.Seats);
}

public static class Seating
{
    public static int? FirstFit(Ledger l, int mask)
    {
        for (var i = 0; i < l.SeatMasks.Length; i++) if ((l.SeatMasks[i] & mask) == 0) return i;
        return null;
    }
    /// Best fit: among seats that fit, the one already most occupied (least free sections left).
    public static int? BestFit(Ledger l, int mask)
    {
        int? best = null; var bestOcc = -1;
        for (var i = 0; i < l.SeatMasks.Length; i++)
        {
            if ((l.SeatMasks[i] & mask) != 0) continue;
            var occ = BitOperations.PopCount((uint)l.SeatMasks[i]);
            if (occ > bestOcc) { bestOcc = occ; best = i; }
        }
        return best;
    }
    public static int? EmptySeat(Ledger l)
    {
        for (var i = 0; i < l.SeatMasks.Length; i++) if (l.SeatMasks[i] == 0) return i;
        return null;
    }
}

public static class Demand
{
    static readonly double[] OriginW = { 0.45, 0.07, 0.12, 0.22, 0.09, 0.05 };
    public const double TerminalShare = 0.60;

    public static int Poisson(Random r, double mean)
    {
        var l = Math.Exp(-mean); var k = 0; var p = 1.0;
        do { k++; p *= r.NextDouble(); } while (p > l);
        return k - 1;
    }
    public static int Destination(Random r, int origin)
    {
        var last = Route.Sections;
        if (origin >= last - 1 || r.NextDouble() < TerminalShare) return last;
        return origin + 1 + r.Next(last - 1 - origin);          // an intermediate stop
    }
    public static Request Draw(Random r)
    {
        var u = r.NextDouble(); var o = 0;
        while (o < OriginW.Length - 1 && u > OriginW[o]) { u -= OriginW[o]; o++; }
        return new Request(o, Destination(r, o));
    }
    public static IReadOnlyList<Request> Trip(Random r, double mean) =>
        Enumerable.Range(0, Poisson(r, mean)).Select(_ => Draw(r)).ToImmutableArray();
}

// ---------------------------------------------------------------- E1: inventory models
public enum Policy { Boolean, Hybrid30, FirstFit, BestFit, LoadOnly }
public sealed record PolicyOutcome(double Revenue, double SeatKm, int Accepted, int Offered, int Fragmented);

public static class E1
{
    public static PolicyOutcome Run(Policy policy, IReadOnlyList<Request> reqs, int seed)
    {
        var rng = new Random(seed);                    // only used by Hybrid30 (channel draw)
        var ledger = Ledger.Empty; var loads = new int[Route.Sections];
        double rev = 0, km = 0; int acc = 0, frag = 0;
        foreach (var q in reqs)
        {
            var mask = Route.Mask(q.From, q.To);
            var viaPortal = policy == Policy.Boolean || (policy == Policy.Hybrid30 && rng.NextDouble() < 0.30);
            var loadOk = Enumerable.Range(q.From, q.To - q.From).All(s => loads[s] < Route.Seats);
            bool ok;
            if (policy == Policy.LoadOnly) ok = loadOk;      // seat numbers assigned only at the end
            else
            {
                var seat = viaPortal ? Seating.EmptySeat(ledger)
                         : policy == Policy.BestFit ? Seating.BestFit(ledger, mask)
                         : Seating.FirstFit(ledger, mask);
                ok = seat is not null;
                if (seat is int s) ledger = ledger.Place(s, viaPortal ? Route.FullMask : mask);
                else if (loadOk) frag++;                     // refused although no section was full
            }
            if (!ok) continue;
            // loads track seats BLOCKED (a portal seat blocks every section)
            for (var s = 0; s < Route.Sections; s++)
                if (viaPortal || (mask & (1 << s)) != 0) loads[s]++;
            rev += Route.Fare(q.From, q.To); km += Route.Km(q.From, q.To); acc++;
        }
        return new PolicyOutcome(rev, km, acc, reqs.Count, frag);
    }

    public static IReadOnlyDictionary<(int, Policy), PolicyOutcome> Sweep(IReadOnlyList<int> means, int trips)
    {
        var d = new Dictionary<(int, Policy), PolicyOutcome>();
        foreach (var mean in means)
            foreach (var p in Enum.GetValues<Policy>())
            {
                var outs = Enumerable.Range(0, trips)
                    .Select(t => Run(p, Demand.Trip(new Random(1000 * mean + t), mean), 7 * t + 1)).ToList();
                d[(mean, p)] = new PolicyOutcome(outs.Average(o => o.Revenue), outs.Average(o => o.SeatKm),
                    (int)Math.Round(outs.Average(o => o.Accepted)), (int)Math.Round(outs.Average(o => o.Offered)),
                    0) with { Fragmented = 0 };
                frag[(mean, p)] = outs.Average(o => (double)o.Fragmented);
                acc[(mean, p)] = outs.Average(o => (double)o.Accepted);
            }
        return d;
    }
    public static readonly Dictionary<(int, Policy), double> frag = new();
    public static readonly Dictionary<(int, Policy), double> acc = new();
}

// ---------------------------------------------------------------- E2: the trip leak
public sealed record World(double AdvanceMean, double PrepaidShare, double NoShow,
    IReadOnlyList<double> WalkOn, double Conceal, double ConcealResold);
public sealed record TripTruth(IReadOnlyList<Pax> Pax, int S, int U, int N, int Cash, int D)
{
    public int Concealed => Cash - D;
    public int NaiveLeak => S - U - D;              // the identity as stated
    public int CorrectedLeak => S - (U - N) - D;    // UPI counted only for seats actually sat in
}

public static class E2
{
    public static World Base(double conceal) => new(38, 0.5, 0.08,
        new[] { 3.0, 1.0, 2.0, 4.0, 2.0, 1.0 }, conceal, Math.Min(1.0, 4 * conceal));

    public static TripTruth Simulate(int seed, World w)
    {
        var r = new Random(seed);
        var ledger = Ledger.Empty; var pax = new List<Pax>();
        foreach (var q in Demand.Trip(r, w.AdvanceMean))
        {
            var mask = Route.Mask(q.From, q.To);
            if (Seating.FirstFit(ledger, mask) is not int seat) continue;
            ledger = ledger.Place(seat, mask);
            var mode = r.NextDouble() < w.PrepaidShare ? Pay.Upi : Pay.Cash;
            pax.Add(new Pax(seat, q.From, q.To, Route.Fare(q.From, q.To), mode, r.NextDouble() >= w.NoShow, false, false));
        }
        var freed = new List<(int Seat, int Mask)>();
        for (var s = 0; s < Route.Sections; s++)
        {
            // rule: a no-show holds the seat until departure from his own stop, then frees downstream
            foreach (var p in pax.Where(p => !p.Shows && p.From == s).ToList())
            {
                var m = Route.Mask(p.From, p.To);
                ledger = ledger.Free(p.Seat, m); freed.Add((p.Seat, m));
            }
            var n = Demand.Poisson(r, w.WalkOn[s]);
            for (var i = 0; i < n; i++)
            {
                var to = Demand.Destination(r, s); var mask = Route.Mask(s, to);
                if (Seating.FirstFit(ledger, mask) is not int seat) continue;
                ledger = ledger.Place(seat, mask);
                var resold = freed.Any(f => f.Seat == seat && (f.Mask & mask) != 0);
                pax.Add(new Pax(seat, s, to, Route.Fare(s, to), Pay.Cash, true, resold, false));
            }
        }
        var final = pax.Select(p => p.Mode == Pay.Cash && p.Shows
            ? p with { Declared = r.NextDouble() >= (p.Resold ? w.ConcealResold : w.Conceal) } : p).ToImmutableArray();
        int S = final.Where(p => p.Shows).Sum(p => p.Fare);
        int U = final.Where(p => p.Mode == Pay.Upi).Sum(p => p.Fare);
        int N = final.Where(p => p.Mode == Pay.Upi && !p.Shows).Sum(p => p.Fare);
        int cash = final.Where(p => p.Mode == Pay.Cash && p.Shows).Sum(p => p.Fare);
        int D = final.Where(p => p.Mode == Pay.Cash && p.Shows && p.Declared).Sum(p => p.Fare);
        return new TripTruth(final, S, U, N, cash, D);
    }

    /// What a rider with a notebook records: each seated passenger is missed with prob `miss`;
    /// otherwise his alighting stop is off by one stop with prob `stopErr`.
    public static int ObservedS(int seed, TripTruth t, double miss, double stopErr)
    {
        var r = new Random(seed); var s = 0;
        foreach (var p in t.Pax.Where(p => p.Shows))
        {
            if (r.NextDouble() < miss) continue;
            var to = p.To;
            if (r.NextDouble() < stopErr)
            {
                var cand = to + (r.Next(2) == 0 ? -1 : 1);
                if (cand > p.From && cand <= Route.Sections) to = cand;
            }
            s += Route.Fare(p.From, to);
        }
        return s;
    }

    public enum Verdict { Build, DoNotBuild, Inconclusive }
    public const double T95 = 1.729;       // one-sided 95%, 19 d.f.
    public static Verdict Decide(IReadOnlyList<double> shares, double theta)
    {
        var n = shares.Count; var mean = shares.Average();
        var sd = Math.Sqrt(shares.Sum(x => (x - mean) * (x - mean)) / (n - 1));
        var half = T95 * sd / Math.Sqrt(n);
        return mean - half > theta ? Verdict.Build : mean + half < theta ? Verdict.DoNotBuild : Verdict.Inconclusive;
    }
}

// ---------------------------------------------------------------- E5: the hold timer
public sealed record HoldOutcome(double Attempts, double LockedInTime, double Rescued, double Orphaned, double BlockedByHold, double TrulyFull);

public static class E5
{
    static double Exp(Random r, double mean) => -mean * Math.Log(1 - r.NextDouble());
    static double Normal(Random r) => Math.Sqrt(-2 * Math.Log(1 - r.NextDouble())) * Math.Cos(2 * Math.PI * r.NextDouble());

    /// Last 90 minutes before departure, `freeSeats` seats still empty, Poisson arrivals.
    public static HoldOutcome Run(int seed, double holdSec, int freeSeats, double arrivals, double abandon)
    {
        var r = new Random(seed);
        var locked = new int[Route.Seats];
        for (var i = freeSeats; i < Route.Seats; i++) locked[i] = Route.FullMask;
        var holds = new List<(int Seat, int Mask, double Expiry)>();
        var events = new PriorityQueue<(int Kind, int Seat, int Mask, double HoldExpiry), double>(); // 0 arrival, 1 webhook
        var t = 0.0; var n = Demand.Poisson(r, arrivals);
        for (var i = 0; i < n; i++) events.Enqueue((0, 0, 0, 0), r.NextDouble() * 5400);
        double att = 0, inTime = 0, rescued = 0, orphan = 0, blocked = 0, full = 0;
        while (events.TryDequeue(out var e, out t))
        {
            holds.RemoveAll(h => h.Expiry <= t);
            if (e.Kind == 0)
            {
                att++;
                var q = Demand.Draw(r); var mask = Route.Mask(q.From, q.To);
                int seat = -1; var freeIgnoringHolds = false;
                for (var i = 0; i < Route.Seats; i++)
                {
                    if ((locked[i] & mask) != 0) continue;
                    freeIgnoringHolds = true;
                    if (holds.Any(h => h.Seat == i && (h.Mask & mask) != 0)) continue;
                    seat = i; break;
                }
                if (seat < 0) { if (freeIgnoringHolds) blocked++; else full++; continue; }
                holds.Add((seat, mask, t + holdSec));
                if (r.NextDouble() < abandon) continue;                       // never pays
                var pay = 45 * Math.Exp(0.8 * Normal(r));                       // lognormal, median 45 s
                var hook = r.NextDouble() < 0.97 ? Exp(r, 3) : Exp(r, 240);     // 3% slow webhooks
                events.Enqueue((1, seat, mask, t + holdSec), t + pay + hook);
            }
            else
            {
                var stillFree = (locked[e.Seat] & e.Mask) == 0
                    && !holds.Any(h => h.Seat == e.Seat && (h.Mask & e.Mask) != 0 && h.Expiry != e.HoldExpiry);
                if (t <= e.HoldExpiry) { locked[e.Seat] |= e.Mask; inTime++; holds.RemoveAll(h => h.Seat == e.Seat && h.Expiry == e.HoldExpiry); }
                else if (stillFree) { locked[e.Seat] |= e.Mask; rescued++; }
                else orphan++;                                                  // debited, seat gone: refund
            }
        }
        return new HoldOutcome(att, inTime, rescued, orphan, blocked, full);
    }
}

public static class Program
{
    static string F(double x, string f = "0") => x.ToString(f, CultureInfo.InvariantCulture);
    static string Pct(double x, string f = "0.0") => (100 * x).ToString(f, CultureInfo.InvariantCulture) + "%";
    static IReadOnlyList<string> Row(params string[] c) => c;

    public static void Main()
    {
        var tables = new List<Table>();

        // ---- T0 fare chart
        tables.Add(new Table("T0", "Illustrative section fare chart (rupees)",
            new[] { "From \\ To" }.Concat(Route.Stops.Skip(1).Select(s => s.Name)).ToArray(),
            Enumerable.Range(0, Route.Sections).Select(i => (IReadOnlyList<string>)
                new[] { Route.Stops[i].Name }.Concat(Enumerable.Range(1, Route.Sections)
                    .Select(j => j > i ? Route.Fare(i, j).ToString() : "")).ToArray()).ToList()));

        // ---- E1
        var means = new[] { 30, 45, 60, 75, 90, 120 };
        const int trips = 4000;
        var e1 = E1.Sweep(means, trips);
        var cap = Route.Seats * Route.Km(0, Route.Sections);
        tables.Add(new Table("E1a", $"Mean fare revenue per trip (rupees), {trips} simulated trips per cell",
            Row("Mean requests", "Boolean seat", "30% portal seats", "Interval first-fit", "Interval best-fit", "Load-only bound", "First-fit vs boolean"),
            means.Select(m => Row(m.ToString(), F(e1[(m, Policy.Boolean)].Revenue), F(e1[(m, Policy.Hybrid30)].Revenue),
                F(e1[(m, Policy.FirstFit)].Revenue), F(e1[(m, Policy.BestFit)].Revenue), F(e1[(m, Policy.LoadOnly)].Revenue),
                "+" + Pct(e1[(m, Policy.FirstFit)].Revenue / e1[(m, Policy.Boolean)].Revenue - 1))).ToList()));
        tables.Add(new Table("E1b", "Seat-kilometre utilisation and refusals",
            Row("Mean requests", "Util. boolean", "Util. first-fit", "Util. load-only", "Refused, boolean", "Refused, first-fit", "First-fit refusals with no section full (per trip)", "Same, best-fit"),
            means.Select(m => Row(m.ToString(), Pct(e1[(m, Policy.Boolean)].SeatKm / cap), Pct(e1[(m, Policy.FirstFit)].SeatKm / cap),
                Pct(e1[(m, Policy.LoadOnly)].SeatKm / cap),
                Pct(1 - E1.acc[(m, Policy.Boolean)] / m), Pct(1 - E1.acc[(m, Policy.FirstFit)] / m),
                F(E1.frag[(m, Policy.FirstFit)], "0.00"), F(E1.frag[(m, Policy.BestFit)], "0.00"))).ToList()));

        // ---- E2 identity
        var conceals = new[] { 0.0, 0.05, 0.10, 0.20, 0.40 };
        const int nTruth = 20000;
        var rowsId = new List<IReadOnlyList<string>>();
        foreach (var k in conceals)
        {
            var ts = Enumerable.Range(0, nTruth).Select(i => E2.Simulate(50_000 + i, E2.Base(k))).ToList();
            double S = ts.Average(t => (double)t.S);
            rowsId.Add(Row(Pct(k, "0"), F(S), F(ts.Average(t => (double)t.U)), F(ts.Average(t => (double)t.N)),
                F(ts.Average(t => (double)t.Cash)), F(ts.Average(t => (double)t.Concealed)),
                Pct(ts.Average(t => (double)t.Concealed) / S), Pct(ts.Average(t => (double)t.NaiveLeak) / S),
                Pct(ts.Average(t => (double)t.CorrectedLeak) / S),
                Pct(ts.Count(t => t.NaiveLeak < 0) / (double)nTruth, "0")));
        }
        tables.Add(new Table("E2a", $"The leak identity against ground truth, per trip (rupees; {nTruth} trips per row)",
            Row("Concealment of ordinary cash", "S sitting fares", "U UPI settled", "N no-show UPI kept", "Cash collected", "True concealed cash C", "C / S", "Stated leak (S−U−D) / S", "Corrected leak / S", "Trips with stated leak < 0"), rowsId));

        // ---- E2 campaign operating characteristic
        var misses = new[] { 0.0, 0.02, 0.05 };
        const int campaigns = 2000; const double theta = 0.03;
        var rowsOc = new List<IReadOnlyList<string>>();
        foreach (var k in conceals)
            foreach (var m in misses)
            {
                var v = new Dictionary<(bool, E2.Verdict), int>();
                double trueShare = 0;
                for (var c = 0; c < campaigns; c++)
                {
                    var naive = new List<double>(); var corr = new List<double>();
                    for (var t = 0; t < 20; t++)
                    {
                        var seed = 9_000_000 + (int)(k * 1000) * 100_000 + c * 20 + t;
                        var tr = E2.Simulate(seed, E2.Base(k));
                        var sHat = E2.ObservedS(seed ^ 0x5bd1e995 + (int)(m * 100), tr, m, 0.05);
                        naive.Add((sHat - tr.U - tr.D) / (double)sHat);
                        corr.Add((sHat - (tr.U - tr.N) - tr.D) / (double)sHat);
                        trueShare += tr.Concealed / (double)tr.S;
                    }
                    foreach (var (isCorr, xs) in new[] { (false, naive), (true, corr) })
                    {
                        var key = (isCorr, E2.Decide(xs, theta));
                        v[key] = v.GetValueOrDefault(key) + 1;
                    }
                }
                string P(bool c, E2.Verdict d) => Pct(v.GetValueOrDefault((c, d)) / (double)campaigns, "0");
                rowsOc.Add(Row(Pct(k, "0"), Pct(trueShare / (campaigns * 20)), Pct(m, "0"),
                    P(false, E2.Verdict.Build), P(false, E2.Verdict.DoNotBuild),
                    P(true, E2.Verdict.Build), P(true, E2.Verdict.DoNotBuild), P(true, E2.Verdict.Inconclusive)));
            }
        tables.Add(new Table("E2b", $"Twenty-trip verdicts at threshold θ = 3% of sitting fares ({campaigns} simulated campaigns per row; alighting stop misread 5% of the time)",
            Row("Concealment of ordinary cash", "True leak C / S", "Passengers missed by rider", "Stated identity: build", "Stated identity: do not build", "Corrected: build", "Corrected: do not build", "Corrected: inconclusive"), rowsOc));

        // ---- E3 head-count coverage
        var ts3 = Enumerable.Range(0, 20000).Select(i => E2.Simulate(700_000 + i, E2.Base(0.10))).ToList();
        double totC = ts3.Sum(t => (double)t.Concealed);
        double Cover(Func<Pax, bool> seen) => ts3.Sum(t => t.Pax.Where(p => p.Shows && p.Mode == Pay.Cash && !p.Declared && seen(p)).Sum(p => (double)p.Fare)) / totC;
        bool Aboard(Pax p, int sec) => p.From <= sec && p.To > sec;
        var rows3 = Enumerable.Range(0, Route.Sections).Select(s =>
            Row($"{Route.Stops[s].Name} → {Route.Stops[s + 1].Name}", Pct(Cover(p => Aboard(p, s)), "0"))).ToList();
        rows3.Add(Row("Two counts: after Arambagh and before Esplanade", Pct(Cover(p => Aboard(p, 3) || Aboard(p, 5)), "0")));
        rows3.Add(Row("Three counts: after Kotulpur, after Arambagh, before Esplanade", Pct(Cover(p => Aboard(p, 2) || Aboard(p, 3) || Aboard(p, 5)), "0")));
        tables.Add(new Table("E3a", "Share of concealed rupees whose passenger is on board when heads are counted (one count per trip, 10% concealment scenario)",
            Row("Section in which heads are counted", "Concealed rupees covered"), rows3));

        var cs = new[] { 0.08, 0.10, 0.15, 0.20 }; var phis = new[] { 0.0, 1.0, 2.0, 5.0 };
        tables.Add(new Table("E3b", "Detection probability q* above which declaring cash pays the conductor better than keeping it, q* = (1 − c) / (1 + φ)",
            new[] { "Commission c" }.Concat(phis.Select(p => $"Penalty φ = {F(p)}× the fare")).ToArray(),
            cs.Select(c => (IReadOnlyList<string>)new[] { Pct(c, "0") }.Concat(phis.Select(p => Pct((1 - c) / (1 + p), "0"))).ToArray()).ToList()));

        // wage neutrality: c1 = c0 + (1 - c0) * C/R ; owner gain = (1 - s)(1 - c0) C
        var tsW = Enumerable.Range(0, 20000).Select(i => E2.Simulate(50_000 + i, E2.Base(0.10))).ToList();
        double R = tsW.Average(t => (double)(t.U + t.Cash)), C = tsW.Average(t => (double)t.Concealed);
        const double c0 = 0.10;
        var rowsW = new[] { 0.0, 0.5, 0.75, 1.0 }.Select(s =>
        {
            var c1 = c0 + s * (1 - c0) * C / R;
            return Row(Pct(s, "0"), Pct(c1, "0.00"), F((1 - s) * (1 - c0) * C * 60), F(-(1 - s) * (1 - c0) * C * 60),
                Pct(-(1 - s) * (1 - c0) * C / (c0 * (R - C) + C)));
        }).ToList();
        tables.Add(new Table("E3c", $"Who gets the recovered leak (10% concealment scenario: ticket money R = ₹{F(R)} and concealed cash C = ₹{F(C)} per trip; commission 10% on declared ticket money; 60 trips a month)",
            Row("Share of recovered cash returned to crew", "Commission that delivers it", "Owner gain per month (₹)", "Crew change per month (₹)", "Crew income change"), rowsW));

        // ---- E4 pricing
        var rows4 = new List<IReadOnlyList<string>>();
        foreach (var m in new[] { 45, 60, 90 })
        {
            double rev = e1[(m, Policy.FirstFit)].Revenue * 60, tickets = E1.acc[(m, Policy.FirstFit)] * 60, prepaid = 0.5;
            foreach (var (flat, per) in new[] { (1000.0, 2.0), (1500.0, 3.0), (2500.0, 5.0) })
            {
                var msgs = tickets * prepaid * 3 * 0.115;
                double Led(double g) => flat + per * tickets * prepaid + msgs + g * rev * prepaid;
                rows4.Add(Row(m.ToString(), F(rev), $"₹{F(flat)} + ₹{F(per)}", F(Led(0)), Pct(Led(0) / rev, "0.00"), Pct(Led(0.02) / rev, "0.00"),
                    F(0.12 * rev * prepaid), Pct(0.12 * prepaid, "0.00")));
            }
        }
        tables.Add(new Table("E4a", "Monthly channel cost for one bus (60 trips, half the tickets prepaid, three utility messages per prepaid ticket)",
            Row("Mean requests", "Fare revenue per month (₹)", "Ledger price: per bus + per prepaid ticket", "Ledger cost, no gateway fee (₹)", "As share of revenue", "With a 2% gateway fee on UPI", "12% commission on the same prepaid seats (₹)", "As share of revenue"), rows4));

        double tk60 = E1.acc[(60, Policy.FirstFit)] * 60 * 0.5;
        tables.Add(new Table("E4b", "The ledger-keeper's own month at ₹1,500 per bus + ₹3 per prepaid ticket, against an assumed ₹25,000 fixed cost (one person at the stand, telephone, hosting)",
            Row("Buses", "Fees collected (₹)", "Less message cost (₹)", "Surplus over assumed cost (₹)"),
            new[] { 5, 8, 15, 30 }.Select(b =>
            {
                var fees = b * (1500 + 3 * tk60); var msg = b * tk60 * 3 * 0.115;
                return Row(b.ToString(), F(fees), F(msg), F(fees - msg - 25000));
            }).ToList()));

        // ---- E5 hold timer
        var rows5 = new List<IReadOnlyList<string>>();
        foreach (var hold in new[] { 60.0, 120, 180, 300, 480, 720 })
        {
            var outs = Enumerable.Range(0, 20000).Select(i => E5.Run(31_000 + i, hold, 10, 25, 0.25)).ToList();
            double a = outs.Sum(o => o.Attempts);
            rows5.Add(Row(F(hold / 60), F(1000 * outs.Sum(o => o.LockedInTime) / a), F(1000 * outs.Sum(o => o.Rescued) / a, "0.0"),
                F(1000 * outs.Sum(o => o.Orphaned) / a, "0.00"), F(1000 * outs.Sum(o => o.BlockedByHold) / a, "0.0"), F(1000 * outs.Sum(o => o.TrulyFull) / a)));
        }
        tables.Add(new Table("E5", "Hold timer, per 1,000 booking attempts (last 90 minutes, 10 seats open, 25 attempts expected, 25% never pay; 20,000 runs per row)",
            Row("Hold (minutes)", "Locked inside the hold", "Paid late, seat still free (locked anyway)", "Paid late, seat gone (refund owed)", "Turned away by an unpaid hold", "Turned away, interval truly sold"), rows5));

        foreach (var t in tables)
        {
            Console.WriteLine($"\n== {t.Id}: {t.Title}");
            Console.WriteLine(string.Join(" | ", t.Headers));
            foreach (var r in t.Rows) Console.WriteLine(string.Join(" | ", r));
        }
        File.WriteAllText("results.json", JsonSerializer.Serialize(tables, new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
    }
}
