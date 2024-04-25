import matplotlib.pyplot as plt
import numpy as np


class AlphaBeta:
    alpha: float
    beta: float

    velocity: np.ndarray
    position: np.ndarray

    def __init__(self, alpha: float, beta: float, initial_position: np.ndarray):
        self.alpha = alpha
        self.beta = beta
        self.position = initial_position
        self.velocity = np.zeros_like(initial_position, dtype=np.float32)

    def update(self, measurement: np.ndarray, dt: float) -> np.ndarray:
        assert dt != 0.0
        predicted_position = self.position + self.velocity * dt
        residual = measurement - predicted_position
        self.position = predicted_position + self.alpha * residual
        self.velocity += (self.beta * residual) / dt
        return self.position

    def get_position(self) -> np.ndarray:
        return self.position

    def predict(self, dt: float) -> np.ndarray:
        self.position += self.velocity * dt
        return self.position


# Raw data to help test the filter and determine alpha/beta
if __name__ == "__main__":
    road_angle = [-146.8643904512811, -146.30993247402023, -147.12709103493012, -146.2149146846536, -146.02345596374315, -145.92280471986928, -145.12467165539783, -144.79984866886764, -144.2933085993971, -143.43690413140592, -143.13010235415598, -142.43140797117252, -141.84277341263095, -141.34019174590992, -141.34019174590992, -140.62240062708972, -141.00900595749454, -141.60483549675396, -141.00900595749454, -140.19442890773482, -138.77228360937983, -138.77228360937983, -139.06336599293303, -136.97493401088198, -136.33221985386965, -137.66300076606714, -135.0, -133.19883948427764, -130.5443971666737, -133.94559549647818, -127.9716037610504, -130.5443971666737, -124.80401175271321, -123.36636600105957, -129.40066066347944, -119.59229635935799, -126.75928817029212, -123.11134196037203, -128.33334010909863, -116.27421215472745, -122.19573393471325, -119.59229635935799, -116.31264849478885, -119.35775354279127, -114.7343025290525, -116.04181659489382, -108.43494882292202, -118.86715946951462, -118.30075576600638, -114.10223450116115, -115.3461759419467, -112.7297320799447, -115.01689347810004, -115.31288380187415, -105.48850144290968, -103.70696100407982, -100.91112838428339, -108.92464441605124, -100.8230112262071, -99.92624550665171, -102.61932229343078, -95.1944289077348, -85.53284093861072, -104.23728046576109, -107.65012421993013, -92.93567344642118, -105.10109816138545, -109.5366549381284, -90.0, -88.51213247117222, -107.61257784292383, -94.63546342690265, -99.86580694308438, -98.13010235415598, -105.25511870305779, -90.0, -104.82647997035568, -89.24615166692924, -84.87180895814716, -103.2405199151872, -103.2405199151872, -82.53943387383165, -96.80905017961341, -105.8323866204222, -80.42577211490818, -101.659292653523, -84.73760460040488, -82.78573796079324, -81.36286858492926, -82.69424046668918, -83.33334010909863, -81.65610841596691, -77.06194368281355, -78.99645914825051, -75.77403610124821, -75.57922687248902, -78.5304696671331, -75.37912601136834, -77.38067770656923, -75.55596427550776, -73.46516214265485, -69.94390526342458, -74.66671520694645, -74.66671520694645, -74.18080605249986, -74.42745640318974, -70.97439396243132, -67.5205656028969, -75.1243179983612, -72.07208023799276, -71.32329826780989, -78.69006752597979, -77.39139320663377, -73.7676493388438, -76.63978155523552, -75.96375653207353, -77.31961650818018, -26.185614475659012, -75.79323388223963, -76.3099503825345, -75.6186054089094, -79.95065141187513, -75.6186054089094, -74.05460409907715, -77.5500034921934, -75.96375653207353, -75.78414652632645, -77.79953127261922, -80.03619581009282, -77.5500034921934, -82.17092348994039, -79.50852298766841, -83.8844964337146, -80.64702074990674, -81.30449712257523, -81.40408856307903, -80.85835977647511, -83.58915997976756, -81.68972283247581, -79.18612248637504, -80.53767779197439, -81.34745820888527, -86.22771639062016, -85.69553103949202, -86.92254460057562, -79.91940201245768, -81.06043499684847, -141.73601121411937, -75.65066795705286, -89.33380002981687, -82.71104767638988, -77.39984017391933, -79.33965170546468, -78.55895613123258, -74.68718163919371, -79.8753283446022, -75.80144597613683, -111.31791227546155, -72.60667785308797, -78.04341575685088, -73.42599224309102, -78.55895613123258, -73.42599224309102, -76.90810693565317, -75.40342447450583, -69.74353782547688, -73.42599224309102, -74.05460409907715, -73.61045966596522, -76.90810693565317, -81.20258929000894, -69.14554196042165, -68.55226367289465, -70.9533757023642, -72.99205304459268, -74.8590161649231, -69.14554196042165, -77.9052429229879, -75.80144597613683, -71.1468412355809, -74.51149855709032, -71.56505117707799, -69.67686317033707, -68.81865049973376, -68.81865049973376, -175.5526151499095, -74.51149855709032, -76.2930389959202, -74.14807184571305, -73.96005669395034, -69.05734945001174, -75.46554491945989, -74.14807184571305, -92.12109639666146, -90.0, -71.3504612459486, -66.37062226934319, -87.82525588538995, -84.99935540244158, -87.8524145717015, -67.58385252065636, -82.8749836510982, -64.01076641616699, -73.49563861824498, -68.19859051364818, -84.28940686250037, -88.58557678859785, -87.13759477388825, -90.0, -67.32865637801915, -92.89857679928171, -94.3432399516894, -92.0700306530411, -82.26640190097714, -71.56505117707799, -75.15454791791618, -87.95459151111278, -88.63607246839707, -79.99202019855866, -74.8590161649231, -74.68718163919371, -71.16156644201924, -76.122470196792, -85.8150838748816, -80.19390724010292, -69.60512391795163, -66.17573941710458, -70.20112364547508, -71.09542415738834, -69.35300917502964, -69.81419699053515, -67.75097634278764, -56.309932474020215, -62.74467162505693, -70.08359400619092, -69.35300917502964, -67.91282839677164, -65.07152586453873, -65.37643521383639, -67.65935204000587, -67.54306100005788, -67.69379494509236, -70.70995378081128, -69.05734945001174, -64.86704500708666, -70.3461759419467, -71.56505117707799, -72.18111108547723, -71.76750891929619, -68.78203042322139, -72.57013718233218, -72.18111108547723, -74.05460409907715, -70.9533757023642, -64.56378361958265, -68.07822140604085, -62.59242456218159, -66.97450799147197, -61.69924423399363, -61.38954033403479, -76.8424572599852, -66.80140948635182, -82.78573796079324, -52.28402365441686, -84.06847307508025, -83.24642596446901, -56.6893691754392, -78.99645914825051, -55.12467165539782, -60.35013649242442, -55.684912400002716, -79.54009090707088, -50.659481840162485, -54.8193006387579, -50.76263288659845, -90.86805144974555, -92.38594403038883, -100.23480276342322, -104.56027561935689, -63.74134044519272, -71.56505117707799, -69.77514056883192, -60.49927537650084, -65.27268561330986, -67.80971110387746, -71.56505117707799, -75.43972438064313, -65.77225468204583, -71.99583839408662, -71.1468412355809, -65.17945866451092, -74.98163936884933, -65.13630342824813, -63.43494882292201, -61.00404048588161, -63.71173787509978, -66.88579201667316, -62.32792177724123, -70.07441934181567, -70.70995378081128, -78.55066201149997, -106.03994330604968, -72.47443162627712, -75.96375653207353, -71.11391263029071, -67.54306100005788, -66.25050550713324, -69.11419853930695, -67.80971110387746, -64.62225613865084, -65.55604521958347, -61.65430637692334, -62.24145939893998, -63.13018691250094, -64.98310652189998, -93.09405805891711, -59.50016676655257, -62.500861268952846, -60.94539590092286, -57.0613094570445, -79.62415507994895, -77.30041551040264, -83.48019824834302, -61.15733986458794, -72.34987578006988, -79.54009090707088, -46.3748347805694, -74.66671520694645, -63.78862198216847, -54.950626687951605, -75.17352002964434, -73.61045966596522, -64.09349200048563, -59.393592968490516, -55.56101069119639, -53.13010235415598, -48.27048792318357, -45.78482460299189, -50.042451069170916, -47.84213788365567, -46.59114027119459, -49.1446237411043, -46.16913932790742, -53.37867250621557, -34.03661035367036, -34.5085229876684, -37.321036192631304, -39.025676713519374, -37.95423087513252, -34.95797636447049, -33.91743053697035, -33.804888573700026, -29.791960809364674, -34.41836447905621, -33.453309454072674, -31.52684188726886, -31.52684188726886, -28.58049325681835, -30.403424474505826, -28.11786972201599, -25.312883801874136, -25.295546400326565, -30.579226872489016, -25.88702626632669, -25.399309570776808, -24.27444113443946, -28.38335414152982, -26.886934192476716, -27.525225743744333, -27.841626069957815, -26.56505117707799, -27.660738511959522, -27.339271342006054, -27.841626069957815, -28.37552185603504, -27.724056235971393, -27.043834847373265, -27.741045707095925, -27.395366663336002, -25.60218755144178, -27.072080237992765, -26.397520135476125, -27.83405394055083, -36.06438615741825, -32.005383208083494, -25.683647180495853, -25.683647180495853, -24.695526290039563, -25.016893478100023, -25.34617594194669, -25.54505604400182, -25.876846684038295, -26.38983527217266, -26.911245027538953, -26.392474049839155, -27.613027823084476, -27.072080237992765, -28.094841901364347, -26.72509483768399, -30.256437163529263, -29.8459319496874, -28.794001311005726, -36.86989764584402, -38.58121326976122, -29.791960809364674, -27.19812794850912, -29.74488129694222, -26.886934192476716, -27.050597007086125, -27.225492005735955, -27.064711005941685, -21.501434324047903, -21.501434324047903, -19.9614035761527, -21.34868667335783, -21.037511025421818, -22.249023657212366, -20.865141570969183, -38.72892255049885, -24.79488411715903, -20.811321755981325, -21.25050550713324, -21.01973386821595, -22.054257710402982, -20.679063344049723, -19.37158112178245, -18.745492012427906, -18.27925415312424, -18.113065807523288, -18.098907048459004, -18.08344538304866, -16.85839876773828, -18.121860247901356, -18.513760024266922, -19.230672375661285, -19.467909201804407, -19.13840872028467, -18.904575842611656, -19.075098492445548, -18.835612078714167, -19.42555522789989, -19.82813377130779, -19.48243426260225, -20.21164850691221, -139.92061603636398, -21.54097591853844, -22.16634582208246, -178.33971763101718, -179.41735114081354, -9.904183212973878, -22.213973225504105, -20.67442476087389, -155.17065341185042, -20.797162279163846, -20.679063344049723, -156.74428509898533, -22.619864948040426, -20.004303627670996, -20.003281091317294, -20.472279519741925, -21.751500283371456, -22.260582216570004, -21.16913877435742, -19.35899417569472, -140.85320152263301, -139.39870535499554, -19.567768581289474, -18.74134044519272, -18.58896936459435, -18.81438552434099, -20.781111658248754, -18.970407808486545, -151.23431708801866, -148.4710119622466, -21.118723982390996, -19.62020964154441, -19.9614035761527, -22.641617331017006, -20.706112397414934, -23.005155125977073, -179.19307054489764, -20.323136829662943, -21.560063906565105, -21.99579645728027, -22.680559607024918, -20.756479412754732, -21.480556475033982, -23.87937136434486, -22.719997281457676, -25.20112364547507, -21.986682177742413, -22.487279909790363, -24.553423083429525, -25.20112364547507, -23.131420814627468, -24.64390046570477, -23.232313746665724, -23.301209710510175, -23.850473908076925, -24.420616378646983, -23.850473908076925]

    wheel_angle = [-42.71450847291565, -40.34998812533832, -39.15725705743409, -35.3164965678341, -33.158195383376125, -33.29572614026015, -30.846054283073002, -32.65022551963203, -32.337849144841165, -33.313045197379914, -26.60471418239494, -26.97363356444621, -26.6993964512, -26.8043480785789, -24.192623732835546, -26.8756590601267, -25.357664289246934, -30.17909623458151, -30.492974226152683, -29.565221143850675, -8.190373103724426, -22.025346641960994, -22.87369946262323, -20.70742088303377, -21.82338561338953, -23.668827088748465, -23.52231687805756, -30.342217837724576, -24.32962287334619, -23.596138137237222, -23.388061159750293, -24.24022933077141, -10.325895878151728, -25.850925945585676, -25.947381255954898, -26.985685774865942, -20.286774418910667, -24.11514638426303, -11.101854221913253, -26.207469719189724, -22.170629287975324, -21.17793216911841, -11.872887503837573, -25.386641359448415, -24.21402107672324, -21.239045630169436, -10.310905967392154, -13.530259265762217, -15.313936593113723, -10.664417075016216, -17.971875519491647, -17.112997607025893, -16.951964018162588, -14.931505481575504, -17.11345823247584, -16.615059257191472, -16.49450342340348, -14.768958813974551, -12.416708858527954, -12.0462052532374, -17.77145047491167, -12.050899537619584, -13.136301739301834, -12.752302076609169, -12.752499118687034, -13.597128753951873, -12.859214327699837, -18.796138541178628, -27.381470913437543, -11.886976967991068, -14.215448656306005, -7.529268470937349, -11.072407035118845, -8.755900578661594, -6.6725450815549605, -3.718139892095773, -17.119661426087767, -10.763535089235742, -6.582817214011336, -14.892583839345646, -16.495824184680515, -16.159331500567834, -4.848734064901341, -2.0755143290040645, -3.519332202151872, 0.6547942199827818, -9.117291086771033, 1.382434981630338, -4.550143765919744, 4.793945167565153, 3.298623722493655, 3.294642979892151, -4.687085425420475, -15.373150672677989, -1.8523110290308717, -3.130077811777839, -5.374332443804271, -1.4730825811319912, -2.8402684137401963, -35.17809828623336, -2.001155971263262, -2.8756966675709634, -1.4478702546901794, -2.3757949374635325, -12.679676806245515, -35.56204863034542, -0.5233597841002681, -17.354941804409133, -10.296242370417305, -0.08481170661510416, 7.321815099477958, 3.6303538953304044, 3.9132767019850334, -37.748322359094324, 6.133224399001298, 5.209725813700986, 8.609602435475583, 9.042057122870744, 3.1120479094272975, 2.2674013384474563, 4.587639692597992, 3.7715283128512977, 4.156762760035655, 8.634505096352664, 3.007321341822553, -7.5796688366228455, 1.7282789590085872, 6.980239671943894, 4.092297339915165, 1.056880405796236, -2.2727787262127843, 0.6259042497812476, -5.217419251981026, 2.133941333637825, -13.293163145861918, 1.9402383042619429, 4.150243450339346, -0.018401304188554315, 4.212225429448275, 0.8865602965352692, 3.492603284900424, 5.026505911454432, 2.1301713136777494, -7.644837034189042, 0.3656416028869821, -36.84339093239914, 1.8745381922592539, 3.553322146452891, 0.5469762271111897, 0.6650512020287138, -38.8414547593966, -2.1798814512036184, 0.5122742919750629, 0.8613529768397516, 3.1407296207947106, -0.9048526148050821, -0.18408100227370824, 4.11118808902818, -2.6631307258973944, -4.2744290241115515, 3.76530208194874, -72.024688157762, -71.97237905626399, -0.07057884244149909, 0.6612299140275084, -0.19857254874467678, -71.30986007309599, 1.3269471369070098, 74.25922291169711, 2.5490774700063747, 69.26804935561267, 2.4079106503085845, 69.4351792911232, 4.59441777667572, 3.2928145391943016, 5.577730394232007, 3.1318165339056194, 3.7341892622316153, 2.5567455039393714, 2.8146340522496622, 6.838759331613484, 76.77464881224698, 7.823456953867901, 2.521589744688744, 5.301761090097359, 1.795021908713794, 3.3741352237778184, 6.959925525979086, 3.310554821571561, 72.26355179571445, 1.4058343923453929, 8.38642694225233, 5.695742546270798, 3.4477802934300894, 6.377766524139019, 0.5109398392199412, 3.4594721931690673, -0.20479456636087562, 1.8326010519761722, -0.6265933256582411, -3.231203797115768, -3.632506647861408, 82.9493570256989, 1.8422658199435868, 69.00907753060855, 77.25423324510739, 0.6825630669715054, -83.67312419054616, 7.159367846445181, 77.91411849409607, 2.643434434249901, 7.8088039231815145, 7.58766557121153, 6.448861621763485, 5.414509966779758, 71.24316465030236, 4.4563954563737305, 3.7501409591984385, 72.02664809326997, 3.825088491301575, 66.97103295795809, 2.351530265738474, 1.030106549413162, -1.15223540062609, 1.7009319424484528, 1.9863892666886829, 3.728225504788519, -1.642870546233244, 77.21404993490496, -3.3022664618107402, 4.005496089045678, 2.7460931437460627, 4.0698566186428, 1.3984884514650586, 4.40355035080308, -78.43869089812299, -0.35060989679887405, 2.906973114112055, 3.682573646609689, 1.7356939353370486, 0.3789679833280464, -0.7832805003366776, 2.127825482942428, 1.1227157338995624, 0.7263558132389982, 71.59930092849082, 3.05227979139487, -1.537794960677138, 2.122540767832553, -75.18118338814523, 0.04265896878944857, 2.880855047485461, 6.096920371896151, 3.5792238247164168, -0.0946983900889845, 72.8595243450293, -4.486535172501379, 83.79368293137954, -3.635520757258275, -1.7908258899705423, 3.9240926693111007, 85.01951929504979, 1.4645507294581932, 7.55415573501646, 85.31604466276217, 6.527824976267033, 6.14015516569241, 84.95252010888609, 8.840495856062738, -72.04284228908558, 8.737804231454431, -75.75457654319152, 3.5512521985197116, 3.443985166539259, 5.167203845356339, 5.167203845356339, -76.32502311964237, 7.508740416296692, 7.037530438586759, 7.283852377104903, 8.047520935365316, -76.82055546686082, 3.1556476149691615, 4.93103187162791, 5.572249482500803, 6.530866842083075, 3.9709056599138877, 4.0741363224051765, 5.422749507770197, 1.8398190754367745, 4.7053111927586695, 72.93542960269242, 0.1262587614259786, -2.548542768003875, -1.4116033995210928, 2.11386822531039, 81.3897234945249, 2.7336882192001757, 0.15054078462201487, 82.87535469715118, 4.792961726087338, 1.4847395156427283, 6.940703584403674, 4.246750711127158, 5.082214542508449, 2.0005818039381302, 4.0761799234905585, 5.63918333954432, 7.931686754624959, 5.542509246857641, 77.26735343044044, 2.2453620736071764, 4.095064162933237, 2.9057822658214585, 7.354700839832637, 2.649311583385493, 3.8816111112305203, -0.725498769526557, 2.686978815863618, 5.4322273033457025, 3.4998601283312896, 78.9772784273444, -17.280535923326788, 8.888345148479338, -69.07039727442142, 0.5161035163419373, -17.18983035725059, 6.972277540536713, 4.980160068004945, 2.353012521120623, 4.309912942511435, 0.7836518444565281, 3.0205570338304706, 71.49432898750483, 2.1747995866224454, 3.2167606415976513, 6.259119615193174, 4.155571712435434, 4.154981601784441, 5.860319736270184, 1.8752119348758112, 4.684528317229458, 1.6366716754981538, 2.049105955372672, 0.5079867986813702, 7.244900772869235, 4.524681920696559, 1.7390663049979198, 9.321645415328073, 10.683731864245399, -77.10260450744656, 4.539422665653355, 2.450202902173073, 8.965947745493105, 5.727230249534146, 83.4622669746921, 1.1948348405953433, 5.567010975268551, 6.797019882093385, 9.980071976904487, 1.2651337941469116, 2.3337852295617245, 3.4334921967014873, 5.6871353284201165, 88.1069460477522, 3.505857047830418, 7.1680256695796984, 4.6945911559012865, 3.2860058171943196, 5.6301318304476515, 3.7212732923653578, -86.58862719030874, 7.537472658159451, -85.71817696651802, 9.98926815624935, 8.706899385379346, -85.13834227257604, 11.18479495532633, 11.124867729176794, 8.7711910690007, 10.84265046443506, 11.287338741821982, 10.128378129071125, 10.307958096565228, 10.755240802073832, 13.522055897963822, 10.97513756497297, 16.446634098321127, 19.498316665178614, 21.536755607871378, 20.48676531220483, 20.78679144502236, 19.34042940370466, 20.77591825523989, 22.07279457164508, 83.6414189760739, 18.0157957772044, -70.38833772900098, 17.351930108200932, 79.44356559627032, 82.37698675527655, 84.29372082168146, 84.85702280320108, 29.075897811174375, 83.86499531803726, 82.42268750301841, 33.56335144149358, 32.68481438227076, 31.36898236435225, 31.741669622446402, 35.34634368731311, 33.506929642376306, 32.72243344800831, 79.54819063485392, 34.3404557326345, 33.25933760625341, 35.37384775557832, 38.189091666146595, -86.88425013227165, 35.25555310793237, 80.96089522265525, 34.730283724843595, 38.66823275619207, 83.53625412640392, 38.8011646261599, 40.024231576056444, 43.94180415348254, 38.35192623264109, 38.71729792854505, 39.177974842516555, 42.699392547739436, 40.331842883453184, 38.24591468854076, 84.13327979030254, 38.3149546493383, 38.80832111717712, 42.118404999691194, 37.92181410815386, 41.84283460001768, 45.20121409571677, 42.06154745982858, 40.69377686601945, 41.482124687476755, 39.55388560824789, 41.0706564918628, 87.00985772166699, 39.33612386632277, 41.21207699044397, 43.283773285134814, 41.01731520982385, 42.28153527311353, 40.35904956170525, 42.03863479547854, 42.46663492794756, 44.014177110742146, 41.33922539078886, 85.49600401903295, 41.65577817797622, 82.21198792003511, 79.17703131498752, 41.11280853892227, 40.067448479807645, 80.85063851643213, 44.65128015528427, 42.8259532683479, 42.55271910950556, 41.247069465497766, 82.8107168443707, 42.82901589365498, 46.30909537690099, 43.69680872166446, 40.969502531282735, 41.17104790628825, 41.02398303875646, -83.89468349842961, 38.558784173144964, 39.278064386622496, 39.910198539655894, 40.48688264512749, 41.79117714780949, 39.67070602078006, -84.44261061751584, 43.51470784297134, 45.84544055058926, 43.70598044553248, 44.37626439838626, 29.995141591577244, 42.15045147009976, 45.10117227911532, 43.10289046480474, 46.25707033689032, 44.82081894222873, 45.063547357704074, 43.39636008143965, 45.503973693420456, 89.56217686210343, 42.938017250685306, 41.23303383219236, 32.111982706650764, 46.39127878400272, 32.97350183467301, -86.25840378603688, 43.7171017456186, 43.48038157585339, 43.830699782837065, -88.24032021025107, 40.372980371555904, 33.75521951930763, 32.863318358861875, -86.64528448419192, 33.19134221682177]

    filtered_wheel = []
    wheel_filter = AlphaBeta(alpha=0.05, beta=0.001, initial_position=wheel_angle[0])
    for angle in wheel_angle:
        wheel_filter.update(angle, 1)
        filtered_wheel.append(wheel_filter.position)

    # plot both the raw and filtered data
    plt.title("Steering Angle")
    plt.plot(wheel_angle, label='Raw')
    plt.plot(filtered_wheel, label='Filtered')
    plt.legend()
    plt.show()

    filtered_road = []
    road_filter = AlphaBeta(alpha=0.06, beta=0.003, initial_position=road_angle[0])
    for angle in road_angle:
        road_filter.update(angle, 1)
        filtered_road.append(road_filter.position)

    # plot both the raw and filtered data
    plt.title("Road Angle")
    plt.plot(road_angle, label='Raw')
    plt.plot(filtered_road, label='Filtered')
    plt.legend()
    plt.show()
