function Wpdf = WalleniusNCHypergeometricpdf(x,n,m,N,omega,accuracy)
%WNChygepdf returns Wallenius' non-central hypergeometric probability density function
% n = number of balls taken
% m = number of red balls
% N = total number of balls in the urn
% omega = odds

if (n < 0 || n > N || m < 0 || m > N || omega < 0)
    error('FSDA:WalleniusNCHypergeometricpdf:Wrginpt',"Parameter out of range in CWalleniusNCHypergeometric");
end

xmin = m + n - N;
% calculate xmin
if (xmin < 0)
    xmin = 0;
end
xmax = n;
%  calculate xmax
if (xmax > m)
    xmax = m;
end


if nargin <6
    accuracy=1e-08;
end

%% Beginning of code

if x<xmin || x>xmax
    if x<xmin
        disp(['x=' num2str(x) ' is smaller than xmin=' num2str(xmin) ' Density is 0'])
    end
    if x>xmax
        disp(['x=' num2str(x) ' is greater than xmax=' num2str(xmax) ' Density is 0'])
    end

    Wpdf=0;
    return
end

if xmin==xmax
    Wpdf=1;
    return
end

if omega==1 % Call the central hypergeometric
    Wpdf=hygepdf(x,N,m,n);
    return
end

x2=n-x;

x0=min([x, x2]);
% em=(x == m || x2 == N-m);

if x0==0 && n>500
    Wpdf=binoexpand(n,x,N,m,omega);
    return
end

% recursive implementation not implemented
% if (double(n)*x0 < 1000 || (double(n)*x0 < 10000 && (N > 1000.*n || em))) {
%      return recursive();
% }

if  (x0 <= 1 && N-n <= 1)
    Wpdf=binoexpand(n,x,N,m,omega);
    return
end


%% FINDPARS
% Beginning of findpars()
xx=[x, n-x];

j=0;
if omega>1
    oo=[1, 1/omega];
else
    oo=[omega, 1];
end
dd=oo(1)*(m-x)+oo(2)*(N-m-xx(2));
d1=1/dd;
% E= (oo(1)*m + oo(2)*(N-m)) * d1;

rr=1;
if rr<=d1
    rr=1.2*d1;
    % Newton-Raphson iteration to find r
end

condexit=0;
while condexit==0
    lastr = rr;
    rrc = 1. / rr;
    z = dd - rrc;
    zd = rrc * rrc;
    for i=[1 2]

        rt = rr * oo(i);
        if (rt < 100.)
            % avoid overflow if rt big
            %r21 = 1-2^rt;
            % r2=2^rt;
            [r21,r2]=pow2_1(rt);

            % pow2_1(rt, &r2);         // r2=2^r, r21=1.-2^r
            a = oo(i) / r21;          %      // omegai/(1.-2^r)
            b = xx(i) * a;            %     // x*omegai/(1.-2^r)
            z  = z+ b;
            zd = zd+ b * a * log(2) * r2;
        end
    end

    if zd == 0
        error('FSDA:WalleniusNCHypergeometricpdf:WrgItr','cannot find r in function CWalleniusNCHypergeometric::findpars');
    end

    rr = rr- z / zd;
    if rr <= d1
        rr = lastr * 0.125 + d1*0.875;
    end
    j=j+1;

    if j== 70
        error('FSDA:WalleniusNCHypergeometricpdf:WrgIter','convergence problem searching for r in function CWalleniusNCHypergeometric::findpars');
    end


    if abs(rr-lastr) < rr * 1e-6
        condexit=1;
    end
end

if omega > 1
    dd = dd*omega;
    rr = rr*oo(2);
end


r = rr;
rd = rr * dd;


% find peak width

ro = r * omega;
if (ro < 300)                      %  avoid overflow
    [k1,~] = pow2_1(ro);
    k1 = -1. / k1;
    k1 = omega*omega*(k1+k1*k1);
else
    k1 = 0;
end

if r < 300                        %  avoid overflow
    [k2,~]= pow2_1(r);
    k2 = -1. / k2;
    k2 = (k2+k2*k2);
else
    k2 = 0.;
end

phi2d = -4.*r*r*(x*k1 + (n-x)*k2);
if (phi2d >= 0.)
    error('FSDA:WalleniusNCHypergeometricpdf:Wrgiter','peak width undefined in function CWalleniusNCHypergeometric::findpars')
    %        /* wr = r = 0.; */

else
    wr = sqrt(-phi2d);
    w = 1/wr;
end

% xLastFindpars = x;

% end of findpars

bico=lnbico(n,x,N,m);

% if (w < 0.04 && E < 10 && (~em || w > 0.004))
%     Wpdf=laplace(r,rd,w,omega,x,n,bico,phi2d,accuracy);
%     return
% end


%% Beginning of integrate
if (w < 0.02 || (w < 0.1 && (x==m || n-x==N-m) && accuracy > 1E-6))
    if accuracy < 1E-9
        s1=0.5;
    else
        s1=1;
    end

    delta = s1 * w;                      % // integration steplength
    ta = 0.5 + 0.5 * delta;
    sumd = integrate_step(1.-ta, ta,rd,r,omega,x,n,bico);      % // first integration step around center peak
    tb=0;
    while (tb<1)
        tb = ta + delta;
        if (tb > 1.)
            tb = 1;
        end

        s  = integrate_step(ta, tb,rd,r,omega,x,n,bico);       % // integration step to the right of peak
        s = s+integrate_step(1.-tb,1.-ta,rd,r,omega,x,n,bico);  %// integration step to the left of peak
        sumd = sumd+s;
        if (s < accuracy * sumd)
            % stop before interval finished if accuracy reached
            break
        end

        ta = tb;
        if (tb > 0.5 + w)
            delta = delta*2;    %  // increase step length far from peak
        end
    end
else
    % difficult situation. Step length determined by inflection points
    sumd=0;
    for t1=[0 0.5]

        t2=t1+0.5;
        tinf = search_inflect(t1, t2,x,n,omega,rd,r);     % find inflection point

        delta = tinf - t1;
        if (delta > t2 - tinf)
            delta = t2 - tinf; % // distance to nearest endpoint
        end

        delta = delta/7;

        if (delta < 1E-4)
            delta = 1E-4;
        end

        delta1 = delta;
        % // integrate from tinf forwards to t2
        ta = tinf;
        tb=0;

        while (tb < t2)
            tb = ta + delta1;
            if (tb > t2 - 0.25*delta1)
                tb = t2; % // last step of this subinterval
            end
            s = integrate_step(ta, tb,rd,r,omega,x,n,bico);        %  // integration step
            sumd = sumd+s;
            delta1 = delta1*2;                 %        // double steplength
            if (s < sumd * 1E-4)
                delta1 = delta1* 8;   %  large step when s small
            end
            ta = tb;

        end

        if tinf
            % // integrate from tinf backwards to t1
            tb = tinf;
            while (ta > t1)
                ta = tb - delta;
                if (ta < t1 + 0.25*delta)
                    ta = t1; % // last step of this subinterval
                end
                s = integrate_step(ta, tb,rd,r,omega,x,n,bico);       % // integration step
                sumd = sumd+ s;
                delta = delta * 2;                       % // double steplength
                if (s < sumd * 1E-4)
                    delta = delta* 8;  % // large step when s small
                end
                tb = ta;
            end
        end
    end
end
Wpdf= sumd*rd;
end

% Auxiliary functions
function [rt, r2]=pow2_1(r)
r2=2^r;
rt=1-r2;
end

function dens=binoexpand(n,x,N,m,omega)
% // calculate by binomial expansion of integrand
%    // only for x < 2 or n-x < 2 (not implemented for higher x because of loss of precision)

if (x > n/2)
    x1 = n-x;
    m1 = N-m;
    m2 = m;
    o = 1/omega;

else
    x1 = x;
    m1 = m;
    m2 = N-m;
    o = omega;
end

if x1 == 0
    yy=FallingFactorial(m2,n);
    xx=FallingFactorial(m2+o*m1,n);
    dens=exp(yy - xx);
elseif  x1 == 1
    q = FallingFactorial(m2,n-1);
    e = o*m1+m2;
    q1 = q - FallingFactorial(e,n);
    e = e-o;
    q0 = q - FallingFactorial(e,n);
    d = e - (n-1);
    dens= m1*d*(exp(q0) - exp(q1));
else
    error('FSDA:WalleniusNCHypergeometricpdf:Wrginpt','x > 1 not supported by function CWalleniusNCHypergeometric::binoexpand');
end
end

function z=FallingFactorial(a,b)
% // calculates ln(a*(a-1)*(a-2)* ... * (a-b+1))

if (b < 30 && round(b) == b && a < 1E10)
    %  direct calculation
    % double f = 1.;
    %for (int i = 0; i < b; i++) f *= a--;
    %return log(f);
    z=log(prod(a:-1:(a-b+1)));

elseif (a > 100*b && b > 1)

    % % // combine Stirling formulas for a and (a-b) to avoid loss of precision
    ar = 1./a;
    cr = 1./(a-b);
    % % // calculate -log(1-b/a) by Taylor expansion
    s = 0;
    n = 1;
    ba = b*ar;
    f = ba;
    lasts=Inf;
    while (s ~= lasts)
        lasts = s;
        s = s+f/n;
        f = f*ba;
        n=n+1;
    end

    z= (a+0.5)*s + b*log(a-b) - b + (1./12.)*(ar-cr);
    %    //- (1./360.)*(ar*ar*ar-cr*cr*cr)

else
    % // use LnFacr function
    lna=LnFacr(a);
    lnab=LnFacr(a-b);
    z=lna - lnab;
end
end



function f=LnFacr(x)
% log factorial of non-integer x
ix = round(x);
if (x == ix)
    f=logfactorial(ix);
    % // x is integer
else
    D=1;
    C0 =  0.918938533204672722;      % // ln(sqrt(2*pi))
    C1 =  1/12;
    C3 = -1/360;
    C5 =  1/1260;
    C7 = -1/1680;
    if (x < 6)
        if (x == 0 || x == 1)
            f=0;
            return
        else
            while (x < 6)
                D = D*x;
            end
        end
    end
    r  = 1. / x;  r2 = r*r;
    f = (x + 0.5)*log(x) - x + C0 + r*(C1 + r2*(C3 + r2*(C5 + r2*C7)));
    if D~=1
        f = f-  log(D);
    end
end
end

function bico=lnbico(n,x,N,m)
x2 = n-x; m2 = N-m;
mFac=logfactorial(m)+logfactorial(m2);
% xLastBico =-99;

% natural log of binomial coefficients.
% returns lambda = log(m!*x!/(m-x)!*m2!*x2!/(m2-x2)!)

xFac = logfactorial(x) + logfactorial(x2) + logfactorial(m-x) + logfactorial(m2-x2);
bico=mFac-xFac;
end

function t=search_inflect(t_from, t_to,x,n,omega,rd,r)
COLORS=2;
rdm1 = rd - 1.;
if (t_from == 0 && rdm1 <= 1.)
    t=0;
    return % //no inflection point
end

rho(1) = r*omega;  rho(2) = r;
xx(1) = x;  xx(2) = n - x;
t = 0.5 * (t_from + t_to);
zeta=zeros(2,3,3);
for i=1:COLORS
    % // calculate zeta coefficients
    zeta(i,1,1) = rho(i);
    zeta(i,1,2) = rho(i) * (rho(i) - 1.);
    zeta(i,2,2) = rho(i) * rho(i);
    zeta(i,1,3) = zeta(i,1,2) * (rho(i) - 2.);
    zeta(i,2,3) = zeta(i,1,2) * rho(i) * 3.;
    zeta(i,3,3) = zeta(i,2,2) * rho(i) * 2.;
end


iter = 0;
sele=repmat([0;0;1;1],20,1);

t1=Inf;
while (abs(t - t1) > 1e-5)
    t1 = t;

    tr = 1. / t;
    log2t = log(t)*(1./log(2));
    % phi(1) = phi(2] = phi[3] = 0.;
    phi=zeros(3,1);
    for i=1:COLORS            % // ca lculate first 3 derivatives of phi(t)
        [q1,q] = pow2_1(rho(i)*log2t);
        q =q/ q1;
        phi(1) = phi(1) -xx(i) * zeta(i,1,1) * q;
        phi(2) = phi(2) - xx(i) * q * (zeta(i,1,2) + q * zeta(i,2,2));
        phi(3) = phi(3) - xx(i) * q * (zeta(i,1,3) + q * (zeta(i,2,3) + q * zeta(i,3,3)));
    end

    phi(1) = phi(1)+rdm1;
    phi(2) = phi(2) - rdm1;
    phi(3) = phi(3) + 2. * rdm1;
    phi(1) = phi(1) *tr;
    phi(2) = phi(2)*tr * tr;
    phi(3) = phi(3)* tr * tr * tr;
    % method = (iter & 2) >> 1;       %  // alternate between the two methods ALDO
    method=sele(iter+1);

    Z2 = phi(1)*phi(1) + phi(2);
    Zd = method*phi(1)*phi(1)*phi(1) + (2.+method)*phi(1)*phi(2) + phi(3);

    if (t < 0.5)
        if (Z2 > 0)
            t_from = t;
        else
            t_to = t;
        end
        if (Zd >= 0)
            % // use binary search if Newton-Raphson iteration makes problems
            % t = (t_from ? 0.5 : 0.2) * (t_from + t_to);    % ALDO
            t= t_from;
        else  % Newton-Raphson iteration
            t = t- Z2 / Zd;
        end

    else
        if (Z2 < 0)
            t_from = t;
        else
            t_to = t;
        end

        if (Zd <= 0)
            % // use binary search if Newton-Raphson iteration makes problems
            t = 0.5 * (t_from + t_to);
        else
            % // Newton-Raphson iteration
            t = t - Z2 / Zd;
        end

    end
    if (t >= t_to)
        t = (t1 + t_to) * 0.5;
    end

    if (t <= t_from)
        t = (t1 + t_from) * 0.5;
    end

    iter=iter+1;

    if iter > 20
        error('FSDA:WalleniusNCHypergeometricpdf:Wrgiter',"Search for inflection point failed in function CWalleniusNCHypergeometric::search_inflect");
    end
end
end


function  s = integrate_step(ta, tb,rd,r,omega,x,n,bico)
IPOINTS=8;

xval=[-0.960289856498,-0.796666477414,-0.525532409916,-0.183434642496,0.183434642496,0.525532409916,0.796666477414,0.960289856498];
weights= [0.10122853629,0.222381034453,0.313706645878,0.362683783378,0.362683783378,0.313706645878,0.222381034453,0.10122853629];

delta = 0.5 * (tb - ta);
ab = 0.5 * (ta + tb);
rdm1 = rd - 1.;
sumd = 0;

for i = 1:IPOINTS
    tau = ab + delta * xval(i);
    ltau = log(tau);
    taur = r * ltau;
    % // possible loss of precision due to subtraction here:
    y = log1pow(taur*omega,x) + log1pow(taur,n-x) + rdm1*ltau + bico;
    if (y > -50)
        sumd = sumd+ weights(i) * exp(y);
    end
end
s=delta * sumd;
end


function z=log1mx(x,~)
z=log(1-x);
end

function z=log1pow(q, x)
% // calculate log((1-e^q)^x) without loss of precision
% z=log((1-exp(q))^x);

if (abs(q) > 0.1)
    y = exp(q);
    y1 = 1 - y;
else
    %  // expand 1-e^q = -summa(q^n/n!) to avoid loss of precision
    y1 = 0;
    qn = 1; i = 1;    ifac = 1;
    y2=Inf;
    while (y1 ~= y2)
        y2 = y1;
        qn = qn*q;  ifac = ifac*i; i=i+1;
        y1 = y1- qn / ifac;
    end
    y = 1. - y1;
end

if (y > 0.1)
    %  // (1-y)^x calculated without problem
    z= x * log(y1);

else
    % // expand ln(1-y) = -summa(y^n/n)
    y1 =1;  i = 1;  z1 = 0;
    z2=inf;
    while (z1 ~= z2)
        z2 = z1;
        y1 = y1*y;
        z1 = z1- y1 / i; i=i+1;
    end
    z =x * z1;
end
end



function [Wpdf]=laplace(r,rd,w,omega,x,n,bico,phi2d,accuracy)


NumSDev= [4.324919041, 4.621231001, 4.900964208, 5.16657812, 5.419983175, 5.662697617, 5.895951217, 6.120756286, 6.337957755, ...
    6.548269368, 6.752300431, 6.950575948, 7.143552034];

% //tables of error function residues
% // 0: precision 1.53E-05
ErfRes=[1.77242680540608204400E+00, 4.42974050453076994800E-01, 5.52683719287987914000E-02, 4.57346771067359261300E-03   ...
    2.80459064155823224600E-04, 1.34636065677244878500E-05, 5.21352785817798300800E-07, 1.65832271688171705300E-08           ...
    4.38865717471213472100E-10, 9.76518286165874680600E-12, 1.84433013221606645200E-13, 2.98319658966723379900E-15           ...
    4.16751049288581722800E-17, 5.06844293411881381200E-19, 5.40629927341885830200E-21, 5.09268600245963099700E-23           ...
    4.26365286677037947600E-25, 3.19120961809492396300E-27, 2.14691825888024309100E-29, 1.30473994083903636000E-31           ...
    7.19567933922698314600E-34, 3.61655672748362805300E-36, 1.66299275803871018000E-38, 7.02143932105206679000E-41           ...
    2.73122271211734530800E-43, 9.81824938600123102500E-46, 3.27125155121613401700E-48, 1.01290491600297417870E-50           ...
    2.92208589554240568800E-53, 7.87247562929246970200E-56, 1.98510836143160618600E-58, 4.69476368999432417500E-61           ...
    1.04339442450396263710E-63, 2.18317315734482557700E-66, 4.30811606197931495800E-69, 8.03081062303437395000E-72           ...
    1.41637813978528824300E-74, 2.36693694351427741600E-77, 3.75309000199992425400E-80, 5.65409397708564003600E-83           ...
    8.10322084538751956300E-86, 1.10610328893385430400E-88, 1.43971150303803736000E-91, 1.78884532267880002700E-94           ...
    2.12393968173898899400E-97, 2.41222807417272408400E-100, 2.62311608532487946600E-103, 2.73362126618952541200E-106;
    %// 1: precision 3.81E-06
    1.77244708953065753100E+00, 4.43074113723358004800E-01, 5.53507546366094128100E-02, 4.60063583541917741200E-03...
    2.85265530531727983900E-04, 1.39934570721569428400E-05, 5.61234181715130108200E-07, 1.87635216633109792000E-08...
    5.29386567604284238200E-10, 1.27170893476994027400E-11, 2.62062404027629145800E-13, 4.66479837413316034000E-15...
    7.22069968938298529400E-17, 9.78297384753513147400E-19, 1.16744590415498861200E-20, 1.23448081765041655900E-22...
    1.16327347874717650400E-24, 9.82084801488552519700E-27, 7.46543820883360082800E-29, 5.13361419796185362400E-31...
    3.20726459674397306300E-33, 1.82784782995019591600E-35, 9.53819678596992509200E-38, 4.57327699736894183000E-40...
    2.02131302843758583500E-42, 8.26035836048709995200E-45, 3.13004443753993537100E-47, 1.10264466279388735400E-49...
    3.62016356599029098800E-52, 1.11028768672354227000E-54, 3.18789098809699663200E-57, 8.58660896411902915800E-60...
    2.17384332055877431800E-62, 5.18219413865915035000E-65, 1.16526530012222654600E-67, 2.47552943408735877700E-70...
    4.97637013794934320200E-73, 9.47966949394160838200E-76, 1.71361124212171341900E-78, 2.94335699587741039100E-81...
    4.80983789654609513600E-84, 7.48676877660738410200E-87, 1.11129798477201315100E-89, 1.57475145101473103400E-92...
    2.13251069867015016100E-95, 2.76249093386952224300E-98, 3.42653604413897348900E-101, 4.07334940102519697800E-104;
    % // 2: precision 9.54E-07
    1.77245216056180140300E+00, 4.43102496776356791100E-01, 5.53772601883593673800E-02, 4.61054749828262358400E-03...
    2.87253302758514987700E-04, 1.42417784632842086400E-05, 5.82408831964509309600E-07, 2.00745450404117050700E-08...
    5.91011604093749423400E-10, 1.49916022838813094600E-11, 3.29741365965300606900E-13, 6.32307780683001018100E-15...
    1.06252674842175897800E-16, 1.57257431560311360800E-18, 2.06034642322747725700E-20, 2.40159615347654528000E-22...
    2.50271435589313449400E-24, 2.34271631492982176000E-26, 1.97869636045309031700E-28, 1.51440731538936707000E-30...
    1.05452976534458622500E-32, 6.70612854853490875900E-35, 3.90863249061728208500E-37, 2.09490406980039604000E-39...
    1.03572639732910843160E-41, 4.73737271771599553200E-44, 2.01016799853191990700E-46, 7.93316727009805559200E-49...
    2.91896910080597410900E-51, 1.00361556207253403120E-53, 3.23138481735358914000E-56, 9.76266225260763484100E-59...
    2.77288342251948021500E-61, 7.41751660051554639600E-64, 1.87191699537047863600E-66, 4.46389809367038823800E-69...
    1.00740435367143552990E-71, 2.15468537440631290200E-74, 4.37372804933525238000E-77, 8.43676369508201162800E-80...
    1.54845094802349484100E-82, 2.70727577941653793200E-85, 4.51412388960109772800E-88, 7.18605932463221426200E-91...
    1.09328719452457957600E-93, 1.59123500193816486400E-96, 2.21770259794482485600E-99, 2.96235081914900644200E-102;
    % // 3: precision 2.38E-07
    1.77245342831958737100E+00, 4.43110438095780200600E-01, 5.53855581791170228000E-02, 4.61401880234106439000E-03...
    2.88031928895194049600E-04, 1.43505456256023050800E-05, 5.92777558091362167400E-07, 2.07920891418090254000E-08...
    6.28701715960960909000E-10, 1.65457546101845217200E-11, 3.81394501062348919800E-13, 7.73640169798996619200E-15...
    1.38648618664047143200E-16, 2.20377376795474051600E-18, 3.11871105901085320300E-20, 3.94509797765438339700E-22...
    4.47871054279593642800E-24, 4.58134444141001287500E-26, 4.23915369932833545200E-28, 3.56174643985755223000E-30...
    2.72729562179570597400E-32, 1.90986605998546816600E-34, 1.22720072734085613700E-36, 7.25829034260272865500E-39...
    3.96321699645874596800E-41, 2.00342049456074966200E-43, 9.40055798441764717800E-46, 4.10462275003981738400E-48...
    1.67166813346582579800E-50, 6.36422340874443565900E-53, 2.26969100679582421400E-55, 7.59750937838053600600E-58...
    2.39149482673471882600E-60, 7.09134153544718378800E-63, 1.98415128824311335000E-65, 5.24683837588056156800E-68...
    1.31326161465641387500E-70, 3.11571024962460536800E-73, 7.01627137211411880000E-76, 1.50162731270605666400E-78...
    3.05816530510335364700E-81, 5.93355048535012188600E-84, 1.09802441010335521600E-86, 1.94008240128183308800E-89...
    3.27631821921541675800E-92, 5.29343480369738200400E-95, 8.19001419434114020600E-98, 1.21456436757992622700E-100;
    % // 4: precision 5.96E-08
    1.77245374525903386300E+00, 4.43112635580628681700E-01, 5.53880993417431935600E-02, 4.61519508177347361400E-03...
    2.88323830371235781500E-04, 1.43956506488931199600E-05, 5.97533121516696046900E-07, 2.11560073234896927000E-08...
    6.49836113541376862800E-10, 1.75091216044688314800E-11, 4.16782737060155846600E-13, 8.80643257335436424800E-15...
    1.65748420791207225100E-16, 2.78707349086274968000E-18, 4.19899868515935354900E-20, 5.68498078698629510200E-22...
    6.93816222596422139400E-24, 7.65747618996655475200E-26, 7.66779861336649418200E-28, 6.98905143723583695400E-30...
    5.81737537190421990800E-32, 4.43568540037466870600E-34, 3.10768227888207447300E-36, 2.00640852664381818400E-38...
    1.19706367104711013300E-40, 6.61729939738396217600E-43, 3.39784063694262711800E-45, 1.62450416252839296200E-47...
    7.24798161653719932800E-50, 3.02428684730111423300E-52, 1.18255348374176440700E-54, 4.34156802253088795200E-57...
    1.49931575039307549400E-59, 4.87879082698754128200E-62, 1.49836511723882777600E-64, 4.34998243416684050900E-67...
    1.19554618884894856000E-69, 3.11506828608539767000E-72, 7.70504604851319512900E-75, 1.81153231245726529100E-77...
    4.05332288179748454100E-80, 8.64127160751002389800E-83, 1.75723563299790750600E-85, 3.41217779987510142000E-88...
    6.33324341504830543600E-91, 1.12470466360665277900E-93, 1.91282818505057981800E-96, 3.11838272111119088500E-99;
    % // 5: precision 1.49E-08
    1.77245382449389548700E+00, 4.43113238150016054000E-01, 5.53888635367372804600E-02, 4.61558298326459057200E-03...
    2.88429374592283566800E-04, 1.44135302457832808700E-05, 5.99599530816354110000E-07, 2.13293263207088596800E-08...
    6.60866899904610148200E-10, 1.80600922150303605400E-11, 4.38957621672449876700E-13, 9.54096365498724593600E-15...
    1.86125270560486321400E-16, 3.26743200260750243300E-18, 5.17322947745786073000E-20, 7.40303709577309752000E-22...
    9.59703297362487960100E-24, 1.12979041959758568400E-25, 1.21090586780714120800E-27, 1.18477600671972569200E-29...
    1.06110784945102789800E-31, 8.72301430014194580800E-34, 6.59978694597213862400E-36, 4.60782503988683505400E-38...
    2.97629996764696360400E-40, 1.78296967476668997800E-42, 9.92947813649120231300E-45, 5.15238281451496107200E-47...
    2.49648080941516617600E-49, 1.13183145876711695200E-51, 4.81083885812771760200E-54, 1.92068525483444959800E-56...
    7.21538203720691761200E-59, 2.55484244329461795400E-61, 8.54021947322263940200E-64, 2.69922457940407460300E-66...
    8.07806757099831088400E-69, 2.29233505413233278200E-71, 6.17627451352383776600E-74, 1.58198519435517862400E-76...
    3.85682833066898009900E-79, 8.96007783937447061800E-82, 1.98575880907873828900E-84, 4.20275001914011054200E-87...
    8.50301055680340658200E-90, 1.64613519849643900900E-92, 3.05222294684008316300E-95, 5.42516704506242119200E-98;
    % // 6: precision 3.73E-09
    1.77245384430261089200E+00, 4.43113402125597019200E-01, 5.53890898808651020700E-02, 4.61570802060252211600E-03...
    2.88466397094702578100E-04, 1.44203545983349722400E-05, 6.00457657669759309400E-07, 2.14076280553580130200E-08...
    6.66287908992827087900E-10, 1.83546080772263722600E-11, 4.51849203153760888400E-13, 1.00053478654150626250E-14...
    2.00133542358651377800E-16, 3.62647881190865840300E-18, 5.96489800325831839200E-20, 8.92069144951359438200E-22...
    1.21499978844978062400E-23, 1.50969159775091919100E-25, 1.71458470816131592700E-27, 1.78354149193378771000E-29...
    1.70298947555869630200E-31, 1.49600537831395400600E-33, 1.21186208172570666700E-35, 9.07362642179266008600E-38...
    6.29382543478586469600E-40, 4.05352760000606626000E-42, 2.42933889358226154400E-44, 1.35768914148821438100E-46...
    7.09017160688256911600E-49, 3.46664168532600651800E-51, 1.58991153690202909500E-53, 6.85218984466549798200E-56...
    2.77986852228382907500E-58, 1.06333492956411188200E-60, 3.84102521375678317000E-63, 1.31221496031384552800E-65...
    4.24584095965170648000E-68, 1.30291378525223696900E-70, 3.79687911940099574200E-73, 1.05205378465263412500E-75...
    2.77502269989758744900E-78, 6.97601832816401403200E-81, 1.67315109709482392200E-83, 3.83268665565667928900E-86...
    8.39358376033290752000E-89, 1.75907817494562062400E-91, 3.53115954806899335200E-94, 6.79562013989671425000E-97;
    % // 7: precision 9.31E-10
    1.77245384925478974400E+00, 4.43113446460012284000E-01, 5.53891560601252504200E-02, 4.61574755288994634700E-03...
    2.88479053368568788400E-04, 1.44228769021976818600E-05, 6.00800544645992949800E-07, 2.14414502554089331400E-08...
    6.68819005926294320800E-10, 1.85032367193584636900E-11, 4.58880445172944815400E-13, 1.02790650461108873560E-14...
    2.09055796622121955200E-16, 3.87357904265687446300E-18, 6.55355746022352119400E-20, 1.01398465283490267200E-21...
    1.43654532753298842400E-23, 1.86580454392148962200E-25, 2.22454554378132065200E-27, 2.43828788210971585600E-29...
    2.46099438567553070000E-31, 2.29136593939231572900E-33, 1.97178483051357608300E-35, 1.57129911859150760300E-37...
    1.16187715309016251400E-39, 7.98791034830625946600E-42, 5.11610271388176540200E-44, 3.05861085454619325800E-46...
    1.71006575230074253400E-48, 8.95787473757552059200E-51, 4.40426750636187741200E-53, 2.03593329808165663200E-55...
    8.86319619094250260800E-58, 3.63949556302483252000E-60, 1.41180525527432472100E-62, 5.18110448656726197600E-65...
    1.80130976146235507900E-67, 5.94089489436009998000E-70, 1.86108901096460881000E-72, 5.54453617603266634800E-75...
    1.57273231131712670500E-77, 4.25229555550383344000E-80, 1.09708064410784368000E-82, 2.70363777400980301400E-85...
    6.37064773173804957600E-88, 1.43666982549400138800E-90, 3.10359876850474266200E-93, 6.42822304267944541900E-96;
    % // 8: precision 2.33E-10
    1.77245385049283445600E+00, 4.43113458380306853400E-01, 5.53891751960330686200E-02, 4.61575984524613369300E-03...
    2.88483285115404915700E-04, 1.44237837119469849000E-05, 6.00933085215778545800E-07, 2.14555059613473259000E-08...
    6.69949807134525424700E-10, 1.85746173246056176400E-11, 4.62510251141501895600E-13, 1.04309449728125451550E-14...
    2.14376794695367282400E-16, 4.03195345507914206800E-18, 6.95901230873262760600E-20, 1.10422005968960415700E-21...
    1.61274044622451622200E-23, 2.17010646570190394600E-25, 2.69272585719737993500E-27, 3.08406442023150341400E-29...
    3.26412756902204044100E-31, 3.19659762892894327800E-33, 2.90079234489442113000E-35, 2.44307440922101839900E-37...
    1.91280099578638699700E-39, 1.39463784147443818800E-41, 9.48568383329895892700E-44, 6.02906080392955580400E-46...
    3.58720420688290561300E-48, 2.00136767763554841800E-50, 1.04877885428425423540E-52, 5.17045929753308956200E-55...
    2.40183088534749939500E-57, 1.05288434613857573000E-59, 4.36191374659545444200E-62, 1.71017740178796946700E-64...
    6.35417287308090154000E-67, 2.24023617204667066100E-69, 7.50388817892399787300E-72, 2.39087016939309798700E-74...
    7.25439736654156264700E-77, 2.09846227207024494800E-79, 5.79315651373498761100E-82, 1.52786617607871741100E-84...
    3.85332605389629328300E-87, 9.30196261538477647000E-90, 2.15126632809118648300E-92, 4.77058936290696223500E-95;
    % // 9: precision 5.82E-11
    1.77245385080234563500E+00, 4.43113461569894215700E-01, 5.53891806760746538300E-02, 4.61576361260268991600E-03...
    2.88484673044866409200E-04, 1.44241019771415521500E-05, 6.00982861902849871600E-07, 2.14611541966231908200E-08...
    6.70435999307504633400E-10, 1.86074527008731886600E-11, 4.64296589104966284700E-13, 1.05109058078120195880E-14...
    2.17373506425627932200E-16, 4.12736258800510237200E-18, 7.22027572389545573000E-20, 1.16641031427122158000E-21...
    1.74261574594878846800E-23, 2.40999131874158664000E-25, 3.08741471404781296800E-27, 3.66622899027160893300E-29...
    4.03832398444680182100E-31, 4.12964092806000764200E-33, 3.92459969957984993300E-35, 3.47023698321199047400E-37...
    2.85870037656881575800E-39, 2.19701222983622897200E-41, 1.57757442199878062800E-43, 1.05998290283581317870E-45...
    6.67461794578944750100E-48, 3.94493775265477963400E-50, 2.19180590286711897200E-52, 1.14647284342367091100E-54...
    5.65409064942635909000E-57, 2.63281413190197920300E-59, 1.15914855705146421000E-61, 4.83173813806023163900E-64...
    1.90931412007029721900E-66, 7.16152712238209948300E-69, 2.55277823724126351900E-71, 8.65775632882397637500E-74...
    2.79685049229469435800E-76, 8.61535752145576873700E-79, 2.53319381071928112300E-81, 7.11686161831786026200E-84...
    1.91227899461300469000E-86, 4.91879425560043181900E-89, 1.21226578717106016000E-91, 2.86511260628508142200E-94;
    %// 10: precision 1.46E-11
    1.77245385087972342800E+00, 4.43113462419744630200E-01, 5.53891822321947835700E-02, 4.61576475266972634100E-03...
    2.88485120632836570100E-04, 1.44242113476668549100E-05, 6.01001089101483108200E-07, 2.14633579957941871400E-08...
    6.70638121912630560800E-10, 1.86219965341716152100E-11, 4.65139560168398521100E-13, 1.05511053035457485150E-14...
    2.18978467579008781700E-16, 4.18179627467181890600E-18, 7.37905600609363562400E-20, 1.20666925770415139000E-21...
    1.83216676939141016100E-23, 2.58616160243870388400E-25, 3.39612594393133643000E-27, 4.15117456105401982300E-29...
    4.72512355800254106200E-31, 5.01108411105699264300E-33, 4.95452692086540934200E-35, 4.57052259669118191500E-37...
    3.93757613394119041600E-39, 3.17143225730425447800E-41, 2.39087136989889684400E-43, 1.68918677399352864600E-45...
    1.11992962513487784300E-47, 6.97720003652956407000E-50, 4.09017183052803247800E-52, 2.25925194899934230000E-54...
    1.17743902383784437300E-56, 5.79751618317805258800E-59, 2.70049127204827368400E-61, 1.19150157862632851000E-63...
    4.98581510751975724600E-66, 1.98102566456273457700E-68, 7.48277410614888503600E-71, 2.68994458637406843000E-73...
    9.21308680313745922900E-76, 3.00957175301701607000E-78, 9.38604174484261857600E-81, 2.79745691952436047200E-83...
    7.97548757616816228000E-86, 2.17700350714256603000E-88, 5.69442820814374326200E-91, 1.42855756885812751800E-93;
    % // 11: precision 3.64E-12
    1.77245385089906787700E+00, 4.43113462645337308000E-01, 5.53891826707801996000E-02, 4.61576509382801447000E-03...
    2.88485262834342722100E-04, 1.44242482379506758200E-05, 6.01007615943023924400E-07, 2.14641957411498484200E-08...
    6.70719685646245707700E-10, 1.86282265411023575000E-11, 4.65522856702499667400E-13, 1.05705070352080171380E-14...
    2.19800647930093079100E-16, 4.21139261151871749000E-18, 7.47068213693802656400E-20, 1.23132525686457329000E-21...
    1.89037080673535316000E-23, 2.70767450402634975900E-25, 3.62208731605653583200E-27, 4.52783644780645903400E-29...
    5.29116794891083221600E-31, 5.78191926529856774600E-33, 5.91019131357709915300E-35, 5.65375339320520942200E-37...
    5.06448494950527399600E-39, 4.25125004489814020300E-41, 3.34702040997479327500E-43, 2.47392597585772167100E-45...
    1.71856809642179370600E-47, 1.12329116466680264100E-49, 6.91635006957699099400E-52, 4.01648185933072044700E-54...
    2.20256743728563483200E-56, 1.14197705850825122000E-58, 5.60474946818590333800E-61, 2.60701847612354797700E-63...
    1.15061401831998511400E-65, 4.82402847794291118400E-68, 1.92339714685666953300E-70, 7.30092195189691915600E-73...
    2.64114863236683700200E-75, 9.11500639536260716600E-78, 3.00399043312000082200E-80, 9.46306767642663343000E-83...
    2.85205432245625504600E-85, 8.23120145271503093200E-88, 2.27678649791096140000E-90, 6.04082678746563674000E-93;
    % // 12: precision 9.09E-13
    1.77245385090390399000E+00, 4.43113462705021723200E-01, 5.53891827935733966800E-02, 4.61576519490408572200E-03...
    2.88485307416075940900E-04, 1.44242604760223605000E-05, 6.01009907022372119900E-07, 2.14645068933581115800E-08...
    6.70751738699247757000E-10, 1.86308168994678478700E-11, 4.65691470353760117700E-13, 1.05795367138350319200E-14...
    2.20205466324054638500E-16, 4.22680889851439179400E-18, 7.52117118137557251000E-20, 1.24569747014608843200E-21...
    1.92626007811754286900E-23, 2.78693040917777943300E-25, 3.77798094465194860200E-27, 4.80270052176922369800E-29...
    5.72806202403284098500E-31, 6.41118455649104110000E-33, 6.73530071235990996000E-35, 6.64287180769401900600E-37...
    6.15272463485746774200E-39, 5.35401292372264035500E-41, 4.37964050507321407500E-43, 3.37013878900376065400E-45...
    2.44151902553507999600E-47, 1.66674472552984171500E-49, 1.07324838386391679300E-51, 6.52532932562465070600E-54...
    3.75007759408864456600E-56, 2.03933010598440151000E-58, 1.05056269424470639500E-60, 5.13240427502016103000E-63...
    2.38044205354512290600E-65, 1.04929890842558070320E-67, 4.40052237815903136000E-70, 1.75760526644875492000E-72...
    6.69249991110777975200E-75, 2.43182093294000139800E-77, 8.44044451319186471300E-80, 2.80086205952805676200E-82...
    8.89407469263960473600E-85, 2.70501913533005623200E-87, 7.88617413146613817400E-90, 2.20568290007963387700E-92];

% // constants for ErfRes tables:
ERFRES_B = 16;        % // begin: -log2 of lowest precision
ERFRES_E = 40;        %// end:   -log2 of highest precision
ERFRES_S =  2;        %// step size from begin to end
ERFRES_N = (ERFRES_E-ERFRES_B)/ERFRES_S+1; %// number of tables
ERFRES_L = 48;       % // length of each table


COLORS = 2;         % // number of colors
MAXDEG = 40;        % // arraysize, maximum expansion degree

omegai= [omega, 1]; % // weights for each color
xi = [x, n-x];   % // number of each color sampled
rho=zeros(2,1);   % [COLORS];           // r*omegai
% double qi;                    // 2^(-rho)
% double qi1;                   // 1-qi
% double qq[COLORS];            // qi / qi1
% double eta[COLORS+1][MAXDEG+1]; // eta coefficients
% double phideri[MAXDEG+1];     // derivatives of phi
% double PSIderi[MAXDEG+1];     // derivatives of PSI
% double * erfresp;             // pointer to table of error function residues

% // variables in asymptotic summation
sqrt8  = 2.828427124746190098; % // sqrt(8)
% double qqpow;                 // qq^j
% double pow2k;                 // 2^k
% double bino;                  // binomial coefficient
% double vr;                    // 1/v, v = integration interval
% double v2m2;                  // (2*v)^(-2)
% double v2mk1;                 // (2*v)^(-k-1)
% double s;                     // summation term
% double sum;                   // Taylor sum

% int i;                        // loop counter for color
% int j;                        // loop counter for derivative
% int k;                        // loop counter for expansion degree
% int ll;                       // k/2
% int converg = 0;              // number of consequtive terms below accuracy
% int PrecisionIndex;           % // index into ErfRes table according to desired precision

% // initialize

phideri=zeros(3,1);
PSIderi= phideri;
qq=zeros(2,1);
eta=zeros(COLORS+1,MAXDEG+1);
%  // find rho[i], qq[i], first eta coefficients, and zero'th derivative of phi

for i=[1 2]
    rho(i) = r * omegai(i);

    if (rho(i) > 40)
        qi=0;
        qi1 = 1;               %   // avoid underflow
    else
        [qi1,qi] = pow2_1(-rho(i));    %    // qi=2^(-rho), qi1=1.-2^(-rho)
    end

    qq(i) = qi / qi1;                 %    // 2^(-r*omegai)/(1.-2^(-r*omegai))
    % // peak = zero'th derivative
    phideri(1) = phideri(1)  + xi(i) * log1mx(qi, qi1);
    % // eta coefficients
    eta(i,1) = 0.;
    eta(i,2) =  rho(i)*rho(i);
    eta(i,3)=eta(i,2);
end

% // r, rd, and w must be calculated by findpars()
% // zero'th derivative
phideri(1) =  phideri(1) - (rd - 1) * log(2);
% // scaled factor outside integral
f0 = rd * exp(phideri(1) + bico );

vr = sqrt8 * w;
phideri(3) = phi2d;

% // get table according to desired precision
% ALDO
PrecisionIndex = floor((-FloorLog2(accuracy) - ERFRES_B + ERFRES_S - 1) / ERFRES_S);

if (PrecisionIndex < 0)
    PrecisionIndex = 0;
end

if (PrecisionIndex > ERFRES_N-1)
    PrecisionIndex = ERFRES_N-1;
end

while (w * NumSDev(PrecisionIndex) > 0.3)
    % // check if integration interval is too wide
    if (PrecisionIndex == 0)
        disp("Laplace method failed. Peak width too high in function CWalleniusNCHypergeometric::laplace");
        break;
    end
    PrecisionIndex=PrecisionIndex-1;                   % reduce precision to keep integration interval narrow
end

erfresp = ErfRes(PrecisionIndex+1,:);        %// choose desired table

degree = MAXDEG;                         % // max expansion degree
if (degree >= ERFRES_L*2)
    degree = ERFRES_L*2-2;
end

% // set up for starting loop at k=3
v2m2 = 0.25 * vr * vr;                   % (2*v)^(-2)
PSIderi(1) = 1.;
pow2k = 8.;
sumd = 0.5 * vr * erfresp(1);
v2mk1 = 0.5 * vr * v2m2 * v2m2;
accur = accuracy * sumd;

% // summation loop
for k = 4:degree
    phideri(k) = 0.;

    % // loop for all (2) colors
    for i = 1:COLORS
        eta(i,k) = 0;
        % // backward loop for all powers
        for j = k:-1:2
            % // find coefficients recursively from previous coefficients
            eta(i,j)  =  eta(i,j)*((j-1)*rho(i)-(k-3)) +  eta(i,j-1)*rho(i)*(j-2);
        end

        qqpow = 1.;

        % // forward loop for all powers
        for j=2:(k+1)
            qqpow = qqpow *qq(i);                 % // qq^j
            % // contribution to derivative
            phideri(k) =  phideri(k) + xi(i) * eta(i,j) * qqpow;
        end
    end

    % // finish calculation of derivatives
    phideri(k) = -pow2k*phideri(k) + 2*(1-k+1)*phideri(k-1);

    pow2k = pow2k *2;  %   // 2^k

    %// loop to calculate derivatives of PSI from derivatives of psi.
    %// terms # 0, 1, 2, k-2, and k-1 are zero and not included in loop.
    %// The j'th derivatives of psi are identical to the derivatives of phi for j>2, and
    %// zero for j=1,2. Hence we are using phideri[j] for j>2 here.
    PSIderi(k) = phideri(k);            %  // this is term # k
    bino = 0.5 * (k-1) * (k-2);         %  // binomial coefficient for term # 3
    for j = 3: k-3       %    // loop for remaining nonzero terms (if k>5)
        PSIderi(k) = PSIderi(k) + PSIderi(k-j) * phideri(j) * bino;
        bino = bino* (k-j)/double(j);
    end

    if mod(k-1,2) == 0  % only for even k
        ll = (k-1)/2;

        s = PSIderi(k) * v2mk1 * erfresp(ll+1);
        sumd = sumd+s;

        % // check for convergence of Taylor expansion
        if (abs(s) < accur)
            converg=converg+1;
        else
            converg = 0;
        end

        if (converg > 1)
            break
        end

        % // update recursive expressions
        v2mk1 = v2mk1*v2m2;
    end
end
%// multiply by terms outside integral
Wpdf= f0 * sumd;
end


function z=FloorLog2(x)
z=floor(log2(x));
end
%FScategory:ProbDist